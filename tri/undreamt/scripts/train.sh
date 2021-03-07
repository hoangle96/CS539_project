python3 train.py  \
--src ./../Sentiment-and-Style-Transfer/data/amazon/sentiment.train.0 \
--trg  ./../Sentiment-and-Style-Transfer/data/amazon/sentiment.train.1 \
--src_embeddings ./../word_embedding/output/wv.vec \
--trg_embeddings ./../word_embedding/output/wv.vec \
--save neg_to_pos  \
--log_interval 10 \
--cuda \
--batch 256 \
--iterations 10000

