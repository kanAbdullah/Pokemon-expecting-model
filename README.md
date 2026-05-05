## Pokémon Legendary Predictor
A machine learning web app that predicts whether a Pokémon is Legendary based on its base stats.

The model uses engineered features on top of the raw stats:

offensive = Attack + Sp. Attack
defensive = Defense + Sp. Defense
balance = standard deviation across all stats
physical_ratio, special_ratio, atk_def_ratio, sp_atk_sp_def_ratio

An image is generated via the FLUX.1-schnell model on Hugging Face Inference API.

[Try yourself!](https://pokemon-expecting-model.vercel.app/)
