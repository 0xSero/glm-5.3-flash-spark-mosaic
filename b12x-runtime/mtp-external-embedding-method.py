# Extracted from the patched pinned Jovian model for pre-build CPU tests.
class Glm5NextMTP:

    def embed_input_ids(self, input_ids: torch.Tensor, multimodal_embeddings: MultiModalEmbeddings | None=None, *, is_multimodal: torch.Tensor | None=None) -> torch.Tensor:
        text_ids = input_ids
        if is_multimodal is not None:
            text_ids = input_ids.masked_fill(is_multimodal.to(device=input_ids.device, non_blocking=True), 0)
        inputs_embeds = self.model.embed_input_ids(text_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        return _merge_multimodal_embeddings(inputs_embeds=inputs_embeds, multimodal_embeddings=multimodal_embeddings, is_multimodal=_require_is_multimodal(is_multimodal))
