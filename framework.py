def get_llm(args):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    # config = AutoConfig.from_pretrained(f"./{args.llm}")
    model = AutoModel.from_pretrained(f"./{args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(f"./{args.llm}")
    return model, tokenizer

class Framework:
    def __init__(self, args):
        from encoder import get_longformer_with_projector
        self.encoder = get_longformer_with_projector(args)
        self.backbone = get_llm(args)

    def forward(self, x):
        text, payload = x
        payload_embeddings = self.encoder(payload)
        if text:
            # TODO: 将文本表征和payload表征对齐
            pass
        else:
            embeddings = payload_embeddings
            #TODO：将embeddings以表格形式呈现
        
        output = self.backbone(embeddings)
        return output

