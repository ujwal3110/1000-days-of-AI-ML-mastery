# 🚀 PHASE 1 
## 📘 Day 01 — Vectors & Dot Product Fundamentals
Building AI From First Principles — One Day at a Time

Welcome to Day 1 of my 1000 Days of AI/ML Mastery.
Phase 1 is all about foundations, but not by reading theory —
👉 I will build the foundations myself from scratch.

Today marks the beginning of implementing a mini-NumPy, a math engine that will later power neural networks, optimizers, attention layers, and full deep learning frameworks.

---

### 🔥 What I Learned Today
- What a vector really represents (geometry, magnitude, direction)
- Why the dot product measures alignment between vectors
(positive = same direction, negative = opposite, 0 = orthogonal)
- Scalar vs vector intuition
- Importance of vector operations in machine learning
(embeddings, gradients, projection, similarity, etc.) 

### 🔍 Deeper Understanding
- Vectors are not just lists of numbers — they are arrows that capture both magnitude and direction, which makes them perfect for representing anything in machine learning that has structure or meaning.
- The dot product becomes powerful because it tells you how much one vector “points in the direction” of another, which is exactly why it can measure similarity, alignment, or influence. When the dot product is positive, the vectors reinforce each other; when negative, they oppose each other; and when zero, they share nothing in common — this is the geometric basis of orthogonality.
- In machine learning, this concept shows up everywhere: embeddings use dot products to measure similarity, gradients use them to update weights, and attention mechanisms in Transformers rely on them to decide which tokens matter to each other. 
- Thinking in vectors helps you visualize models as systems moving through high-dimensional space, not just crunching numbers. Once this intuition becomes second nature, understanding deeper concepts like projections, cosine similarity, and attention becomes effortless

---
### 🛠️ What I Built Today

#### 📁 File
**Directory:** `foundations/math/`  
**File:** `vector.py` (example)

#### 🔧 Implemented Functions

- `dot(a, b)`
- `add(a, b)`
- `sub(a, b)`
- `scale(a, α)`


###### ➡️ This is the first building block for matrix multiplication, neural networks, and gradients.

#### 🧪 Experiments

- Tested dot product with different vector orientations
- Verified geometric interpretation using angle formula
- Created a small notebook to plot 2D vectors and visualize dot product sign

Screenshot of notebook added to:

experiments/day1_vector_experiments.ipynb

#### 🎯 Plan for Tomorrow

Implement vector operations:

- scalar multiply
- vector subtraction
- magnitude + normalization
- Finish complete Vector API

Start implementing Matrix class on Day 4

#### 🌅 Daily Reflection

Today sets the foundation for the next 1000 days.
A simple vector is the seed from which LLMs, Transformers, and GPUs grow.
Building from first principles is the only way to become a true AI engineer.







