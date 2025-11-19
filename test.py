from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer
import torch.nn as nn
import torch
from fire import Fire
from accelerate import dispatch_model

def main(use_long_prompt: int = 0, use_xformers: int = 0, eval_mode: int = 0, auto_dispatch: int = 1):
    use_long_prompt = bool(use_long_prompt)
    use_xformers = bool(use_xformers)
    eval_mode = bool(eval_mode)
    auto_dispatch = bool(auto_dispatch)
# 加载模型和分词器
    model_name_or_path = "./Qwen3-VL-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if auto_dispatch:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, trust_remote_code=True, device_map="auto")
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.visual = nn.Identity()
    if eval_mode:
        model = model.eval()
    else:
        model = model.train()

    device_map = {
        "model.language_model.embed_tokens": "cuda:0",
        "model.language_model.norm": "cuda:0",
        "lm_head": "cuda:0",
        "model.visual": "cuda:0",
        "model.language_model.rotary_emb": "cuda:0",
        # Layers 0-17 on device 0
        **{f"model.language_model.layers.{i}": "cuda:0" for i in range(0, 18)},
        # Layers 18-35 on device 1
        **{f"model.language_model.layers.{i}": "cuda:1" for i in range(18, 36)},
    }
    if not auto_dispatch:
        model = dispatch_model(model, device_map=device_map)
    
    # model = dispatch_model(model, device_map="auto")

    # 构造简单输入
    short_prompt = "你好，Qwen3！请介绍一下你自己。"
    long_prompt = (
        "你好，Qwen3-VL-8B-Instruct！现在我想通过一个非常长的文本来测试你的处理能力。"
        "以下是一段包含丰富内容的长篇文字，用于考察你的理解与生成效果："
        "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在研究、开发用于模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。"
        "随着信息时代的到来，AI 已经渗透到社会生活的方方面面，从自动驾驶、医疗诊断、金融风控、智能家居，到自然语言处理、机器翻译、图像识别等领域，无一不体现出人工智能技术的巨大影响力。"
        "人工智能的发展离不开数据的积累与计算力的提升。大数据为AI模型的训练提供了丰富的资源，而先进的硬件，如GPU、TPU、FPGA等则加速了模型的推理和部署。"
        "目前，深度学习作为人工智能的核心技术之一，在语音识别、计算机视觉和自然语言处理等诸多应用场景中取得了突破性的成果。"
        "例如，基于Transformer结构的预训练大模型如GPT、BERT、T5等，大大推动了NLP领域的发展。"
        "与此同时，AI的伦理与安全话题同样受到广泛关注，包括数据隐私保护、算法偏见、可解释性及决策透明度等都成为学术界与产业界探讨的重点。"
        "进一步地，随着AI技术的普及，许多国家与地区也开始制定相应的法律法规，规范人工智能的发展与应用，确保其造福社会。"
        "人工智能不仅是一项技术，更是一种赋能工具，正在推动各行各业发生深刻变革。例如，在教育领域，智能助教与个性化学习系统可以辅助教师因材施教，提高学生的学习效率。"
        "在医疗健康行业，AI能够辅助医生进行疾病筛查、药物研发、医学影像分析等，提升诊疗水平和服务效率。同时，工业制造领域借助AI实现了设备预测性维护、柔性制造和生产流程优化，大大节约了成本。"
        "值得一提的是，AI在艺术创作、游戏设计、社会治理等新兴领域同样展现了强大的创新潜力。"
        "针对自然语言处理的进一步发展，大型语言模型具备了更强的知识记忆、推理判断和多轮对话能力。它们不仅可以回答各种类型的问题，还能参与文章撰写、摘要生成、内容创作和代码编写等多项任务。"
        "目前，诸如OpenAI的ChatGPT，Google的Bard，百度的文心一言，以及你Qwen3-VL-8B-Instruct等模型，都已成为智能助手和行业应用的重要底座。"
        "然而，AI的商业化进程依然面临诸多挑战，包括通用性与专业性的平衡、算力资源投入、应用落地的个性化适配，以及环境与社会责任的考量。"
        "期待未来在全社会的共同努力下，人工智能能够以更加安全、可靠、智能的形态，服务大众，助力创新，造福人类。"
        "以上便是一段超长测试字符串，其长度已超过1000字，用于验证你的长文本处理能力。请基于上述内容介绍你自己、讨论人工智能相关的话题，并生成合理的文本。"
        "从全球的角度来看，人工智能技术的竞争日趋激烈，各国都在积极布局AI战略，推动本土创新发展，这也促使跨国合作与交流变得更加重要。"
        "与此同时，随着人工智能在交通、农业、能源、物流等传统行业的深度融合，产业数字化转型步伐明显加快。例如，智慧交通系统通过AI实现拥堵预测与路线优化，智能农业则利用图像识别和数据分析提升作物产量和病虫害防治效率。"
        "在环境保护方面，AI已被应用到气候监测、垃圾分类、资源调度等领域，助力实现可持续发展目标。"
        "此外，随着AI生成内容（AIGC）的快速发展，虚拟人、数字艺术、自动化内容创作等新形态不断涌现，推动了创意产业的革新。"
        "AI对人类社会的影响愈加深远，这不仅仅体现在生产力的提升，更涉及到社会结构、就业形态、教育体系等诸多方面。例如，AI技术推动了远程办公、在线教育和数字医疗的发展，使得优质资源能够更广泛地惠及更多群体。"
        "尽管如此，我们也需警惕技术滥用带来的风险，如深度伪造、虚假信息泛滥、就业技能鸿沟扩大等问题。"
        "因此，社会各界需共同努力，开展跨学科研究，制定完善的伦理规范，加强对AI的治理和监管，推动技术向善。"
        "对于未来，人工智能将不仅仅是工具，更可能成为人类智慧的延伸和合作伙伴。或许，随着技术的不断进化，我们终将迎来人机共融、智慧共生的崭新时代。"
        "你的能力覆盖多模态理解与生成任务，包括文本、图像等数据的综合处理。请用专业且通俗的语言，结合上述丰富场景，详细介绍你自己擅长的方向，以及你对人工智能未来发展的看法和展望。"
        "请仔细分析当前AI技术的瓶颈，提出创新设想，并以通用用户和开发者的视角给出针对应用落地的具体建议。"
        "结合实际案例说明大型语言模型、感知系统与推理机制的结合在现实生产生活中的价值，同时就人机协作、智能服务等议题进行探索。"
        "最后，请展望一下AI未来在知识发现、科学研究、社会治理等更广泛领域的革命性潜力——你的回答应尽可能全面、深入，并展示Qwen3-VL-8B-Instruct的强大能力。"
        "目前，人工智能还在持续进化，尤其是在跨模态交互、人本协同以及自主决策等方面取得显著进展。智能体（Agent）系统结合感知、认知、推理和规划等多重能力，正在驱动智能机器人和虚拟助手的广泛部署。"
        "随着强化学习、因果推断等技术的发展，AI不仅能够在仿真环境中自我成长，也逐步具备迁移能力，更加贴合实际场景的应用需求。例如，自动驾驶汽车集成了多传感器融合、实时控制和路径规划等功能，极大拓展了交通系统的智能边界。"
        "此外，联邦学习、隐私计算等方案正在解决数据孤岛与隐私保护的问题，推动AI在医疗、金融等敏感领域的落地应用。国内外顶尖高校和企业不断投入大量资源，打造开源生态，提升算法算法创新能力及工具链完善度，共同加速AI技术成果转化。"
        "在科研和教育领域，AI正支撑知识图谱、自动化科研、智能实验等新型模式，实现文献推理、数据挖掘和跨领域协作等诸多突破，为科学探索赋能。"
        "智能内容生成也正在为文化创意产业提供无限可能，从智能剧本、配乐到数字世界的虚拟人物建设，多样化的AI应用正在重塑人们的表达方式和创作体验。"
        "展望未来，类脑智能、群体智能等前沿方向或将成为人工智能突破现有架构及算力瓶颈的新路径。AI如何与道德伦理、法律诉求和人类核心价值观和谐共生，是全球必须面对的挑战。"
        "Qwen3-VL-8B-Instruct等大型模型的持续升级，为多语言、多模态、多任务的智能交互提供了坚实基础。希望你能给出对AI社会责任的思考，如何用算法助力弱势群体、消弭信息壁垒，促进包容性科技发展。"
        "请分析你在低资源环境下的适用性、可扩展性及落地实践，并阐述如何与人类建立互信关系，共建透明、可控、可解释的智能系统。"
        "最后，期待你对人机协同共创的未来愿景、AI对绿色可持续发展的贡献，以及在宇宙深空探索等极端环境中的应用潜能进行全面展望。"
    )
    if use_long_prompt:
        prompt = long_prompt
    else:
        prompt = short_prompt
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "image",
    #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    #             },
    #             {"type": "text", "text": "Describe this image."},
    #         ],
    #     }
    # ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    from preprocess.utils import _ids_to_str
    import sys
    input = _ids_to_str(inputs.input_ids[0], type="qwen3vl")

    if use_xformers:
        from transformers import AttentionInterface
        from use_xformers_gqa import xformers_gqa_attention_forward
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        AttentionInterface.register("sdpa", sdpa_attention_forward)
        AttentionInterface.register("xformers_gqa", xformers_gqa_attention_forward)
        model.set_attn_implementation("xformers_gqa")

    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # 重置峰值内存计数器
    # print(inputs)
    # INSERT_YOUR_CODE
    # 将inputs.input_ids转为inputs_embeds, 并提取attention_mask和position_ids
    # 假定inputs包含input_ids, attention_mask, position_ids（如果没有则生成），模型和tokenizer已加载
    input_ids = inputs["input_ids"]
    input_embeddings_module = model.get_input_embeddings()
    inputs_embeds = input_embeddings_module(input_ids)
    attention_mask = inputs.get("attention_mask", None)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).expand(3, -1).unsqueeze(0)
    print(inputs_embeds.shape, attention_mask.shape, position_ids.shape)
    sys.stdout.flush()
    output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=512,  # 可根据需要调整
                do_sample=False,  # 使用贪心解码
            )
            
    # output = model.generate(**inputs, max_new_tokens=1280, do_sample=False)
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"峰值GPU内存使用量: {peak_memory:.2f} MB")
    result = tokenizer.decode(output[0], skip_special_tokens=False)

    print("Input: ", input)
    print("Output: ", result)

if __name__ == "__main__":
    Fire(main)