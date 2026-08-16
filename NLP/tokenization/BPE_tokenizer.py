from collections import Counter;

class BPETokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size= vocab_size;
        self.vocab={};
        self.merges= {};

    def train(self, text):
        # create initial vocabulary from characters
        words= text.lower().split();

        word_tokens =[];

        for word in words:
            tokens = list(word) + ["</w>"];
            word_tokens.append(tokens);

        # build initial vocabulary
        self.vocab = set();

        for tokens in word_tokens:
            self.vocab.update(tokens);

        # repeatedly merge the most frequent pair
        while len(self.vocab) < self.vocab_size:
            pair_counts = Counter();

            for tokens in word_tokens:
                for i in range(len(tokens) -1):
                    pair = (tokens[i], tokens[i+1]);
                    pair_counts[pair] +=1;

            if not pair_counts:
                break;

            best_pair, frequency = pair_counts.most_common(1)[0];

            # stop if no useful pair exists
            if frequency < 2:
                break;

            new_token = "".join(best_pair);

            self.merges[best_pair]= new_token;
            self.vocab.add(new_token);

            # apply the merge
            new_word_tokens = [];

            for tokens in word_tokens:
                merged =[];
                i =0;

                while i < len(tokens):
                    if (
                        i < len(tokens) -1
                        and (tokens[i] , tokens[i+1]) == best_pair
                    ): 
                        merged.append(tokens[i])
                        i +=2;
                    else:
                        merged.append(tokens[i]);
                        i += 1;

            word_tokens = new_word_tokens;

        # assign IDs
        self.token_to_id ={
            token: idx 
            for idx, token in enumerate(sorted(self.vocab))
        }

        self.id_to_token ={
            idx: token
            for token, idx in self.token_to_id.items()
        }

    def encode(self, text):
        words = text.lower().split();

        token_ids =[];

        for word in words:
            tokens = list(word) + ["</w>"];

            # apply learned merges
            for pair, merged_token in self.merges.items():
                new_tokens =[];
                i =0;

                while i < len(tokens):
                    if (
                        i < len(tokens) -1
                        and (tokens[i] , tokens[i+1])== pair
                    ):
                        new_tokens.append(merged_token)
                        i +=2;
                    else: 
                        new_tokens.append(tokens[i])
                        i += 1;
                tokens = new_tokens;

            for token in tokens:
                if token in self.token_to_id:
                    token_ids.append(self.token_to_id[token]);
        
        return token_ids;





    def decode(self, token_ids):
        tokens = [
            self.id_to_token[token_id]
            for token_id in token_ids
        ]

        text = "".join(tokens)

        return text.replace("</w>", " ")


text = """
low lower lowest
low lower lowest
low lower lowest
"""

tokenizer = BPETokenizer(vocab_size=30)

tokenizer.train(text)

print(tokenizer.vocab)

text = "lowest"

tokens = tokenizer.encode(text)

print(tokens)