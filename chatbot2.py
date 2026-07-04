import os
from threading import Thread
import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from openai import OpenAI

# ==========================================
# النماذج المتاحة (مساران):
#   1) Qwen3.5-2B → عبر Transformers، يدعم وضع التفكير (thinking) القابل للتبديل.
#   2) GPT-5      → عبر الـ OpenAI API.
# النموذج المحلي يُحمّل كسولاً عند أول استخدام لتوفير الذاكرة عند الإقلاع.
# ==========================================
QWEN_MODEL = "Qwen/Qwen3.5-2B"
GPT_LABEL  = "GPT-5 (OpenAI API)"

# حدود الرموز (Tokens):
#   - رسالة المستخدم الواحدة: 512 رمزاً كحد أقصى (يُقصّ الزائد).
#   - كامل سياق المحادثة المُرسل للنموذج: 4096 رمزاً (نُبقي الأحدث).
MAX_MSG_TOKENS  = 1024
MAX_CHAT_TOKENS = 4096

# ميزانيتان منفصلتان للتوليد في وضع التفكير:
#   - THINK_BUDGET : أقصى عدد رموز يُسمح به لكواليس التفكير. إن نفدت دون أن يُغلق
#                    النموذج التفكير (</think>) نُغلقه قسراً وننتقل لتوليد الجواب.
#   - ANSWER_BUDGET: ميزانية مستقلة للجواب، فلا يبتلع التفكيرُ حصة الجواب أبداً.
# بهذا نضمن وصول جواب فعلي حتى لو أطال النموذج في التفكير.
THINK_BUDGET  =512
ANSWER_BUDGET =512
THINK_END_ID  = 151668   # رمز </think> في مُرمِّز Qwen3

_qwen_models = {}   # سجلّ نماذج Qwen المحمّلة عبر Transformers

# ------------------------------------------
# تحميل Qwen عبر Transformers (bfloat16 على الـ CPU لتقليل ذروة الذاكرة)
# ------------------------------------------
def get_qwen_model(model_id):
    if model_id not in _qwen_models:
        print(f"⏳ جاري تحميل نموذج {model_id} ...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if torch.cuda.is_available():
            # على الـ GPU: تحميل عادي بالدقة التلقائية
            mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
        else:
            # على الـ CPU: نحمّل بدقة bfloat16 بدون تكميم int8.
            # السبب: تكميم float32→int8 يُحدث ذروة ذاكرة ضخمة (~10-11GB لنموذج 2B) تسبب OOM
            # على الـ Space المجاني. تحميل bfloat16 يُبقي الذروة عند ~نصف ذلك فتتّسع النماذج الأكبر،
            # مقابل توليد أبطأ قليلاً على الـ CPU (لا تسريع int8).
            print("🔻 تحميل بدقة bfloat16 على CPU لتقليل ذروة استهلاك الذاكرة...")
            mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        _qwen_models[model_id] = (tok, mdl)
    return _qwen_models[model_id]

# ==========================================
# توليد Qwen عبر Transformers مع دعم وضع التفكير القابل للتبديل
# Qwen3 يوصي: وضع التفكير temp=0.6/top_p=0.95 — الوضع العادي temp=0.7/top_p=0.8 — وtop_k=20 دائماً
# ==========================================
def _truncate_text_tokens(tok, text, max_tokens):
    """يقصّ النص إلى max_tokens رمزاً كحد أقصى (نُبقي البداية).
    نتجاهل ما ليس نصاً (None/غير str) تماماً كما يفعل قالب المحادثة، تفادياً لخطأ المُرمِّز."""
    if not isinstance(text, str) or not text:
        return text
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tok.decode(ids[:max_tokens], skip_special_tokens=True)

def qwen_generate(messages, enable_thinking, tok, mdl, max_new_tokens=1024):
    temperature = 0.6 if enable_thinking else 0.7
    top_p = 0.95 if enable_thinking else 0.8
    device = next(mdl.parameters()).device

    # حدّ رسالة المستخدم: نقصّ كل رسالة مستخدم إلى MAX_MSG_TOKENS رمزاً قبل بناء البرومبت
    messages = [dict(m) for m in messages]
    for m in messages:
        if m["role"] == "user":
            m["content"] = _truncate_text_tokens(tok, m["content"], MAX_MSG_TOKENS)

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    model_inputs = tok([text], return_tensors="pt").to(device)

    # حدّ سياق المحادثة: إن تجاوز الإدخال MAX_CHAT_TOKENS نُبقي آخر (الأحدث) رموزه فقط
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    if input_ids.shape[-1] > MAX_CHAT_TOKENS:
        input_ids = input_ids[:, -MAX_CHAT_TOKENS:]
        attention_mask = attention_mask[:, -MAX_CHAT_TOKENS:]

    generated_ids = mdl.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=20,
    )
    return generated_ids[0][len(input_ids[0]):].tolist()

# ==========================================
# معيار إيقاف: نوقف التوليد فور إنتاج رمز نهاية التفكير </think>
# ==========================================
class _StopOnToken(StoppingCriteria):
    def __init__(self, token_id):
        self.token_id = token_id
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1].item() == self.token_id

# ==========================================
# صياغة العرض: نغلّف التفكير في صندوق قابل للطي ثم نُلحق الجواب
# ==========================================
def _compose_output(badge, thinking, answer):
    if thinking:
        return f"{badge}<details><summary>🧠 كواليس تفكير النموذج</summary>\n\n{thinking}\n\n</details>\n\n{answer}"
    return f"{badge}{answer}"

# دالة مساعدة: تُشغّل التوليد على خيط منفصل وتبثّ النص، وتُعيد تسلسل الرموز الكامل
def _run_stream(mdl, tok, gen_kwargs):
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(gen_kwargs, streamer=streamer)
    box = {}
    def _run():
        box["out"] = mdl.generate(**gen_kwargs)
    thread = Thread(target=_run, daemon=True)
    thread.start()
    for new_text in streamer:
        yield ("chunk", new_text)
    thread.join()
    yield ("done", box.get("out"))

# ==========================================
# نسخة بثّية من توليد Qwen بميزانيتين منفصلتين (تفكير/جواب).
# في وضع التفكير: مرحلة 1 تولّد التفكير حتى </think> أو نفاد THINK_BUDGET،
# ثم نُغلق التفكير قسراً ونشغّل مرحلة 2 لتوليد الجواب بميزانية ANSWER_BUDGET
# مستقلة — فلا يبتلع التفكيرُ حصةَ الجواب أبداً.
# ==========================================
def qwen_generate_stream(messages, enable_thinking, tok, mdl, badge):
    temperature = 0.6 if enable_thinking else 0.7
    top_p = 0.95 if enable_thinking else 0.8
    device = next(mdl.parameters()).device

    # حدّ رسالة المستخدم: نقصّ كل رسالة مستخدم إلى MAX_MSG_TOKENS رمزاً قبل بناء البرومبت
    messages = [dict(m) for m in messages]
    for m in messages:
        if m["role"] == "user":
            m["content"] = _truncate_text_tokens(tok, m["content"], MAX_MSG_TOKENS)

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    model_inputs = tok([text], return_tensors="pt").to(device)

    # حدّ سياق المحادثة: إن تجاوز الإدخال MAX_CHAT_TOKENS نُبقي آخر (الأحدث) رموزه فقط
    input_ids = model_inputs.input_ids
    attention_mask = model_inputs.attention_mask
    if input_ids.shape[-1] > MAX_CHAT_TOKENS:
        input_ids = input_ids[:, -MAX_CHAT_TOKENS:]
        attention_mask = attention_mask[:, -MAX_CHAT_TOKENS:]

    sampling = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=20,repetition_panalty=1.05)

    # ---- الوضع العادي (بلا تفكير): مرحلة واحدة لتوليد الجواب ----
    if not enable_thinking:
        raw = ""
        for kind, payload in _run_stream(mdl, tok, dict(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=ANSWER_BUDGET, **sampling,
        )):
            if kind == "chunk":
                raw += payload
                yield _compose_output(badge, "", raw.strip("\n"))
        yield _compose_output(badge, "", raw.strip("\n") or "⚠️ لم يصل أي جواب.")
        return

    # ---- وضع التفكير: مرحلة 1 — توليد التفكير حتى </think> أو نفاد الميزانية ----
    think_out = None
    think_raw = ""
    for kind, payload in _run_stream(mdl, tok, dict(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=THINK_BUDGET,
        stopping_criteria=StoppingCriteriaList([_StopOnToken(THINK_END_ID)]),
        **sampling,
    )):
        if kind == "chunk":
            think_raw += payload
            # نعرض التفكير مباشرة داخل الصندوق أثناء توليده (الجواب لم يبدأ بعد)
            thinking_live = think_raw.split("</think>")[0].replace("<think>", "").strip("\n")
            yield _compose_output(badge, thinking_live, "")
        else:
            think_out = payload

    thinking_text = think_raw.split("</think>")[0].replace("<think>", "").strip("\n")

    # ---- المرحلة 2: نضمن إغلاق التفكير ثم نولّد الجواب بميزانية مستقلة ----
    gen_ids = think_out
    if THINK_END_ID not in gen_ids[0].tolist():
        # نفدت ميزانية التفكير دون إغلاق → نُغلقه قسراً بإلحاق </think>
        close = tok("</think>\n\n", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        gen_ids = torch.cat([gen_ids, close], dim=-1)
    attn2 = torch.ones_like(gen_ids)

    answer_raw = ""
    for kind, payload in _run_stream(mdl, tok, dict(
        input_ids=gen_ids, attention_mask=attn2,
        max_new_tokens=ANSWER_BUDGET, **sampling,
    )):
        if kind == "chunk":
            answer_raw += payload
            yield _compose_output(badge, thinking_text, answer_raw.strip("\n"))

    answer = answer_raw.strip("\n") or "⚠️ لم يكتمل الجواب ضمن حدّ التوليد. جرّب رفع ANSWER_BUDGET أو إعادة المحاولة."
    yield _compose_output(badge, thinking_text, answer)

# ==========================================
# تحميل مسبق للنموذج عند بدء التشغيل بدل التحميل الكسول
# نحمّل Qwen مقدّماً حتى لا يتأخر أول ردّ للمستخدم.
# (تبقى دالة get_qwen_model كما هي فتجد النموذج محمّلاً في سجلّه)
# ==========================================
def preload_models():
    print("🚀 تحميل مسبق للنموذج (Qwen)...")
    get_qwen_model(QWEN_MODEL)
    print("✅ اكتمل التحميل المسبق للنموذج.")

# ==========================================
# حدود الرموز لمسار OpenAI: نعدّ الرموز بـ tiktoken، ومع بديل تقريبي إن لم تتوفّر
# ==========================================
try:
    import tiktoken
    _OAI_ENC = tiktoken.get_encoding("o200k_base")   # ترميز نماذج GPT الحديثة
except Exception:
    _OAI_ENC = None

def _oai_truncate(text, max_tokens):
    """يقصّ نصاً إلى max_tokens رمزاً (بـ tiktoken، أو تقدير ~4 أحرف/رمز كبديل)."""
    if not isinstance(text, str) or not text:
        return text
    if _OAI_ENC is None:
        return text if len(text) <= max_tokens * 4 else text[: max_tokens * 4]
    ids = _OAI_ENC.encode(text)
    if len(ids) <= max_tokens:
        return text
    return _OAI_ENC.decode(ids[:max_tokens])

def _oai_count(text):
    if not isinstance(text, str) or not text:
        return 0
    return len(_OAI_ENC.encode(text)) if _OAI_ENC is not None else max(1, len(text) // 4)

def _oai_fit_messages(messages, max_tokens):
    """يقصّ كل رسالة مستخدم إلى MAX_MSG_TOKENS، ويُبقي المحادثة ضمن max_tokens
    عبر إسقاط أقدم الرسائل مع الحفاظ على رسالة النظام."""
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m["role"] == "user":
            m["content"] = _oai_truncate(m["content"], MAX_MSG_TOKENS)
    system = msgs[0] if msgs and msgs[0]["role"] == "system" else None
    body = msgs[1:] if system else msgs
    total = lambda: (_oai_count(system["content"]) if system else 0) + sum(_oai_count(m["content"]) for m in body)
    while total() > max_tokens and len(body) > 1:
        body.pop(0)   # نُسقط الأقدم (نُبقي الأحدث)
    return ([system] if system else []) + body

# ==========================================
# تنظيف رسائل المساعد القديمة من الشارات/كواليس التفكير قبل تمريرها للنموذج
# ==========================================
def _strip_assistant_extras(content):
    if not isinstance(content, str):
        return ""
    if "</details>" in content:                                  # إزالة صندوق التفكير (Qwen)
        content = content.split("</details>")[-1].strip()
    if content.startswith("🟢") and "\n\n" in content:           # إزالة شارة المسار
        content = content.split("\n\n", 1)[-1].strip()
    return content

# ==========================================
# الدالة الأساسية لمعالجة المحادثة بنظام الـ Messages الحديث
# (غلاف يلتقط أي خطأ ويعرضه نصاً في الشات بدل شارة Gradio العامة "Error")
# ==========================================
def chat_engine(user_input, chat_history, model_choice, enable_thinking, api_key, system_prompt):
    try:
        yield from _chat_engine_impl(user_input, chat_history, model_choice, enable_thinking, api_key, system_prompt)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "assistant", "content": f"❌ خطأ عام في المعالجة:\n\n`{type(e).__name__}: {str(e)}`"})
        yield chat_history, ""

def _chat_engine_impl(user_input, chat_history, model_choice, enable_thinking, api_key, system_prompt):
    if not user_input.strip():
        yield chat_history, ""
        return

    # 1. تهيئة مصفوفة الرسائل ببرومبت النظام (System Prompt)
    messages = [{"role": "system", "content": system_prompt}]

    # 2. تفكيك التاريخ القديم بصيغة الـ Dicts الجديدة
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            content = _strip_assistant_extras(content)
        messages.append({"role": role, "content": content})

    # 3. إضافة رسالة المستخدم الجديدة الحالية
    messages.append({"role": "user", "content": user_input})

    # 4. تقليم التاريخ: نُبقي آخر تبادل واحد فقط (سؤال+جواب) حتى لا يتضخّم السياق
    MAX_HISTORY_MSGS = 4
    if len(messages) >  MAX_HISTORY_MSGS :
        messages = [messages[0]] + messages[-(MAX_HISTORY_MSGS + 1):]

    # ------------------------------------------
    # المسار الأول: نموذج Qwen3 المحلي
    # ------------------------------------------
    if model_choice == QWEN_MODEL:
        badge = "🟢 *ردّ مباشر من Qwen*\n\n"
        # نضيف رسالة المستخدم ورسالة مساعد فارغة نُحدّثها تدريجياً مع تدفّق الرموز
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": badge})
        try:
            tok, mdl = get_qwen_model(model_choice)
            # البثّ: كل دفعة نُحدّث آخر رسالة مساعد ونُنتج الحالة الجديدة للعرض الحيّ
            for final_output in qwen_generate_stream(messages, enable_thinking, tok, mdl, badge):
                chat_history[-1]["content"] = final_output
                yield chat_history, ""
        except Exception as e:
            # نُظهر الخطأ الفعلي بدل رسالة Gradio العامة حتى يسهل تشخيصه
            import traceback
            traceback.print_exc()
            chat_history[-1]["content"] = f"❌ خطأ أثناء تشغيل النموذج المحلي ({model_choice}):\n\n`{type(e).__name__}: {str(e)}`"
            yield chat_history, ""
        return

    # ------------------------------------------
    # المسار الثاني: GPT-5 عبر الـ API
    # ------------------------------------------
    elif model_choice == GPT_LABEL:
        chat_history.append({"role": "user", "content": user_input})
        if not api_key.strip():
            chat_history.append({"role": "assistant", "content": "⚠️ يرجى إدخال الـ API Key أولاً."})
            yield chat_history, ""
            return
        # رسالة مساعد فارغة نُراكم عليها دفعات البثّ
        chat_history.append({"role": "assistant", "content": ""})
        try:
            # نطبّق حدّي الرموز: 512 لكل رسالة مستخدم و4096 لكامل السياق
            messages = _oai_fit_messages(messages, MAX_CHAT_TOKENS)
            client = OpenAI(api_key=api_key)
            # stream=True: نستقبل الردّ على شكل دفعات ونعرضها فور وصولها
            stream = client.chat.completions.create(
                model="gpt-5", messages=messages, max_tokens=1024, stream=True
            )
            acc = ""
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    acc += delta
                    chat_history[-1]["content"] = acc
                    yield chat_history, ""
            if not acc:
                chat_history[-1]["content"] = "⚠️ لم يصل أي محتوى من الـ API."
                yield chat_history, ""
        except Exception as e:
            chat_history[-1]["content"] = f"❌ خطأ: {str(e)}"
            yield chat_history, ""
        return

# ==========================================
# بناء واجهة المستخدم (Gradio UI Layout)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate")) as demo:
    gr.Markdown("# 🤖 منصة المحادثة الذكية متعددة النماذج")

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### ⚙️ إعدادات النظام")
            system_prompt_input = gr.Textbox(
                label="توجيهات النظام (System Prompt)",
                value="أنت مساعد ذكي ومرح، تجيب باختصار ودقة، وتتحدث باللغة العربية المفهومة.",
                lines=4,
                placeholder="اكتب تعليمات الموديل هنا..."
            )
            model_dropdown = gr.Dropdown(choices=[QWEN_MODEL, GPT_LABEL], value=QWEN_MODEL, label="اختر النموذج")
            thinking_checkbox = gr.Checkbox(label="🧠 تفعيل وضع التفكير (Qwen فقط)", value=True)
            api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
            clear_button = gr.Button("🧹 مسح الذاكرة")

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="شاشة الشات", height=550)
            with gr.Row():
                user_msg = gr.Textbox(placeholder="اكتب سؤالك هنا واضغط Enter...", scale=4)
                submit_button = gr.Button("إرسال 🚀", variant="primary", scale=1)

    submit_event = submit_button.click(
        chat_engine,
        inputs=[user_msg, chatbot, model_dropdown, thinking_checkbox, api_key_input, system_prompt_input],
        outputs=[chatbot, user_msg]
    )
    user_msg.submit(
        chat_engine,
        inputs=[user_msg, chatbot, model_dropdown, thinking_checkbox, api_key_input, system_prompt_input],
        outputs=[chatbot, user_msg]
    )
    clear_button.click(fn=lambda: [], outputs=chatbot, queue=False)

if __name__ == "__main__":
    # تحميل النماذج مقدّماً قبل رفع الواجهة حتى يكون أول ردّ سريعاً.
    preload_models()
    # نفعّل الطابور (queue) حتى يعمل البثّ التدريجي للدوال المُولّدة (generators)
    demo.queue()
    # على Colab نحتاج share=True لإنشاء رابط عام؛ على Hugging Face Space يجب ألا نفعّله.
    on_hf_space = os.environ.get("SPACE_ID") is not None
    demo.launch(share=not on_hf_space)
