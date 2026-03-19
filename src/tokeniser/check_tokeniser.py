import sentencepiece as spm

'''
if it's sentence-piece model use this code
'''
spm_model = "tokenizer_bpe_4k/rna_bpe.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model)

s = "AUGGCUAGCUAG"
print("SP pieces:", sp.encode_as_pieces(s)[:30])
print("SP ids:", sp.encode_as_ids(s)[:30])
print("SP unk_id:", sp.unk_id())