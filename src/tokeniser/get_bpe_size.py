import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("tokenizer_bpe_4k/rna_bpe.model")

print("SentencePiece piece size:", sp.get_piece_size())
print("First 10 pieces:", [sp.id_to_piece(i) for i in range(10)])