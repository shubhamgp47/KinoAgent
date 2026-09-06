# FilmGPT

app.py and backend.py have tools for my letterboxd and tmdb

app_local_llm and abackend_local_llm have integreated local ollama llm

oscar tools files have oscar awards data tool included

watchlist_tool files are on top of these


##Semantic questions 
Mood/theme-based discovery: "What are some dark, slow-burn psychological thrillers I've rated highly?" — there's no exact metadata field for "slow-burn" or "dark," so the model needs to infer this from synopsis text.

"Films similar to X": "Suggest films similar to Oldboy" — similarity is inherently a distance metric, which is what embeddings compute directly.

Vague recall: "What's that movie I watched about a guy stuck in a time loop?" — you're describing plot elements, not a title, so semantic matching against synopses is ideal.

Taste pattern analysis: "What kind of films do I usually enjoy?" — this benefits from your chat_node system prompt's multi-tool guidance, pulling top-rated films via personaltaste_retriever then checking their common genres/themes via synopsis_retriever.

Cross-collection reasoning: "Have I rated any war films highly, and if so, what do critics generally say about their themes?" — combines your subjective ratings with TMDb's descriptive data.

Ambiguous or partial descriptions: "A surreal film with multiple timelines that confused me" — this only works because embeddings capture semantic closeness to descriptive language, not exact keywords.

Poor Fits (Where You Hit the Ceiling You Just Found)
These are categorical/exhaustive questions, which is exactly the Nolan case:

"Have I watched all of [director]'s films?" — needs exact filtering, not ranking.

"How many horror films have I rated 5 stars?" — needs a count over an exact filter, not top-N similarity.

"List every film I've watched from the 1990s." — a date range filter, not semantic.

"Which actor appears most often in my watched films?" — an aggregation, which vector search fundamentally can't do.