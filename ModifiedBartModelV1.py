#%% Imports

import torch

from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader

from tqdm import tqdm

#%% Class

class ModifiedBartModelV1():
    def __init__(self, model_name:str ="facebook/bart-base", device='cuda:0'):

        if 'cuda' in device:
            if torch.cuda.is_available():
                self.device = torch.device(device)
            else:
                print('Cuda not availabile, using CPU.')
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.model_name = model_name
        self.model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def get_decoder_layer_outputs(self, input_ids, attention_mask, decoder_input_ids):
        """Retrieve outputs from each decoder layer of BART."""
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,  # Enables returning hidden states
            return_dict=True
        )

        decoder_hidden_states = outputs.decoder_hidden_states  # Tuple of hidden states from each layer
        return decoder_hidden_states

    def generate_summary(self, text, layer_number=-1, max_length=512):
        """Generate a summary by iteratively sampling tokens from logits."""
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)

        summary_ids = []
        for _ in range(max_length):

            decoder_layer_outputs = self.get_decoder_layer_outputs(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )

            next_token_logits = self.model.lm_head(decoder_layer_outputs[layer_number])[:, -1, :]

            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            summary_ids.append(next_token_id.item())

            # Correctly append next token to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)

        summary = self.tokenizer.decode(summary_ids, skip_special_tokens=True)
        return summary, inputs

    def fine_tune_lm_head(self, train_dataset, layer_number, epochs=50, batch_size=32, learning_rate=2e-5):
        """Fine-tune the lm_head for a specific decoder layer."""
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        # Freeze all model parameters except the lm_head
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

        # Set up optimizer and loss function
        optimizer = torch.optim.AdamW(self.model.lm_head.parameters(), lr=learning_rate)

        # Tokenize the training dataset
        def tokenize_function(examples):
            inputs = self.tokenizer(
                examples['article'],
                max_length=1024,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    examples['highlights'],
                    max_length=128,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
            inputs['labels'] = labels['input_ids']
            return inputs

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_dataloader = DataLoader(tokenized_train, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        progress_bar = tqdm(range(epochs), desc=f'Layer {layer_number}')
        for epoch in progress_bar:
            # progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1} (Layer {layer_number})')
            if epoch % 5 == 0:
                learning_rate *= 0.8
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Get decoder layer outputs
                with torch.no_grad():
                    decoder_layer_outputs = self.get_decoder_layer_outputs(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        decoder_input_ids=batch['labels'][:, :-1]  # Shifted labels as decoder input
                    )

                # Use the specific decoder layer's output
                layer_output = decoder_layer_outputs[layer_number]

                # Compute logits and loss
                logits = self.model.lm_head(layer_output)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.model.config.vocab_size),
                    batch['labels'][:, 1:].reshape(-1)  # Shifted labels as targets
                )

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({'loss': loss.item()})

        torch.save(self.model.lm_head.state_dict(), f'lm_head_{layer_number}.pth')

# %%
