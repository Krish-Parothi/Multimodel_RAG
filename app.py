'''
Step-1: PDF Document --> Extract Text and images from PDF

Step-2: from the text that was extracted, we will convert them in to chunks and we will performs embeddings.

Step 3: from images that extracted we will perform embeddings.

Now the best idea is we will use single model i.e CLIP(Provided by OpenAI and it is OpenSource) to handle both text and Image Embeddings at once. This Model is best for text-Image mapping.

The Reason we are using CLIP , this model is trained on 400 million images

CLIP : Contrastive Language Image Pre-Training. (Vision Transformers + Transformers)
-> In case of images it uses Vision tranformers.
-> In case of text it uses Text transformers.

✅ 1) Image Embeddings kya hote hain?

Embedding = image ka meaning / features ko numbers (vectors) ki list mein convert karna.

Example:
Ek image → model → output:
[0.12, -0.88, 0.34, 0.91, ...]

Yeh hundreds/thousands numbers hote hain — image ka semantic meaning represent karte hain.

👉 Embedding = sirf numerical values, not the image itself.

✅ 2) Base64 kya hota hai?

Base64 = image file ko text-string (A-Z, a-z, 0-9, +, /) mein convert karna,
taaki:
JSON me bheja ja sake
DB me rakhna easy ho
Network me safe transfer ho

👉 Base64 = image ka text version, NOT embedding.
❗ Most important clarity
Embedding ≠ Base64
They are totally different things:
Concept	Kya karta hai?
Embedding	Image ka meaning ko numbers mein convert karta hai
Base64	Image file ko text string bana deta hai





⭐ CLIP ka use kya hai? (Very Simple Explanation)
CLIP = Ek model jo images aur text dono ko embeddings (numbers) me convert karta hai — aur dono same space me hota hai.

Iska matlab:

Image → numbers

Text → numbers
Aur dono embeddings mutual meaning ko represent karte hain.

Isliye:

"dog" text ka embedding

dog ki image ka embedding

Ek doosre ke kareeb honge.

⭐ CLIP kyu use hota hai?
1) Images ko understand karne ke liye (semantic meaning nikalne ke liye)

CLIP model image ko dekhkar uska “meaning” samajhta hai.

Example:
Dog ki image → embedding me features:

fur

face

animal

pet

4 legs

Numbers me represent ho jata hai.

2) Text-image matching ke liye

CLIP ka biggest feature:

👉 Text aur images ka meaning compare ho sakta hai.

Tum query likho:

“brown dog playing”

CLIP embedding compare karega ki kaunsi image ke numbers is text ke numbers se zyada match karte hain.

Isliye CLIP:

image search

visual RAG

captioning

recommendation
me use hota hai.

3) RAG me CLIP kahan use hota hai?

Visual RAG me steps:

Image ko CLIP se embedding banao.

Text queries ko bhi CLIP se embedding banao.

Vector database me store karo (image embeddings).

Jab koi query aati hai:

Query ko CLIP text encoder se numbers me convert karo

Similar embeddings ki images retrieve karo

Isliye CLIP bridge hai between image & text meaning.

⭐ 4) Base64 ke context me CLIP ka role

Base64 sirf image data ko text string banata hai.
CLIP ke liye zaruri steps:

Base64 decode → image file → CLIP → embedding numbers

CLIP khud Base64 se koi kaam nahi karta.
CLIP sirf processed image pe kaam karta hai.

⭐ CLIP ka main kaam ek line me

CLIP image aur text ko same-dimensional meaning vectors me convert karta hai, jisse hum images ko text se search kar sakein.

Agar chaho to main ek diagram bana kar full flow bhi dikhado:
Image → Base64 → decode → CLIP → Embedding → Vector DB → Text Query → CLIP → Match







USE VECTOR STORE FOR Storing Embeddings (Vector Store that we will be using is --> FAISS Vector Store. )


Take a New Query we will again use CLIP Model to convert this query into embeddings.

Then we will do vector search after Embedding so this is called as retreivers.

Before Sending it to LLM Model(OpenAI or GPT-4.1 etc) it should be in a specific format.

Then We wil get Multimodal answe.


We will use PYMuPDF for text and image extraction

'''