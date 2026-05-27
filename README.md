# Quranic Guidance Recommendation System

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20404414.svg)](https://doi.org/10.5281/zenodo.20404414)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Run Online (No Installation Required)

[![Kaggle Part 1](https://img.shields.io/badge/Kaggle-TF--IDF%20Recommendation-blue)](https://www.kaggle.com/code/shagufakhan/quran-tfidf-recommendation)
[![Kaggle Part 2](https://img.shields.io/badge/Kaggle-Semantic%20Search-blue)](https://www.kaggle.com/code/shagufakhan/quran-semantic-search)

**Part 1 — TF-IDF Recommendation:**
https://www.kaggle.com/code/shagufakhan/quran-tfidf-recommendation

**Part 2 — Semantic Search:**
https://www.kaggle.com/code/shagufakhan/quran-semantic-search

Click the badges above to run either pipeline interactively
on Kaggle without any local setup or installation.

> **Disclaimer:** This system is for spiritual reflection and educational
> reference only. It does not constitute religious rulings (fatwas).
> All recommendations are based on well-established Quranic themes
> and authentic hadith sources cited with evidence levels.

## Overview
An AI-powered Quranic verse and surah recommendation system that
suggests relevant surahs, key verses, and authentic duas based on
a user's emotional state, situation, or spiritual need. Built using
two complementary NLP approaches — TF-IDF content-based filtering
and Sentence Transformer semantic search.

## Two Parts

### Part 1 — TF-IDF Content-Based Recommendation (Intermediate)
- User types a situation or selects a category
- TF-IDF vectorization matches keywords to database entries
- Cosine similarity ranks surahs, verses, and duas
- NLP keyword lexicon detects emotional context from natural language

### Part 2 — Sentence Transformer Semantic Search (Advanced)
- User types naturally in any phrasing
- all-MiniLM-L6-v2 model encodes meaning into 384-dimensional vectors
- Semantic similarity finds relevant guidance even without exact keywords
- Outperforms TF-IDF on indirect and metaphorical expressions

## Database Contents
| Content | Count | Details |
|---|---|---|
| Surahs | 20 | With Arabic names, themes, evidence levels |
| Key Verses | 15 | Arabic text + English translation + hadith source |
| Duas | 8 | Arabic + transliteration + translation + source |
| Categories | 20 | Anxiety to Ramadan to Marriage to Travel |

## Evidence Level System
| Level | Meaning |
|---|---|
| A | Sahih (authentic) hadith — strongest evidence |
| B | Quranic theme — from surah content and classical tafsir |
| C | Scholarly consensus |

## Sample Recommendations
**Input:** "I feel so worried and anxious about my future"
**Detected:** Anxiety
Surahs  : Ash-Sharh, Ta-Ha, An-Nas
Verses  : Al-Inshirah 94:5-6, Al-Baqarah 2:286, Ar-Ra'd 13:28
Duas    : Dua for anxiety (Sahih Bukhari 6369)

**Input:** "My chest feels tight and I cannot breathe easy"
**Detected:** Anxiety (semantic — no exact keywords)
Surahs  : Ash-Sharh, Al-Fatihah, An-Nas
Verses  : Al-Baqarah 2:286, Al-Inshirah 94:5-6, Ta-Ha 20:25-26
Duas    : Dua for anxiety (Sahih Bukhari 6369)

## Categories Covered
| Domain | Categories |
|---|---|
| Emotional | Anxiety, Sadness, Fear, Loneliness, Depression, Hope |
| Physical | Illness, Pain, Healing, Recovery |
| Worship | Friday, Ramadan, Tahajjud, Dhikr, Night Prayer |
| Personal | Patience, Gratitude, Forgiveness, Repentance |
| Life Events | Marriage, Travel, Exams, Financial Difficulty, Death |

## Output Figures
| Figure | Description |
|---|---|
| fig1_category_coverage.png | Coverage across all 20 categories |
| fig2_evidence_quality.png | Evidence level distribution |
| fig3_category_overlap.png | Category overlap heatmap |
| fig4_recommendation_scores.png | TF-IDF scores for anxiety query |
| fig5_full_dashboard.png | Full recommendation dashboard |
| fig6_nlp_detection.png | NLP category detection results |
| fig7_nlp_summary_table.png | NLP results summary table |
| figA1_query_similarity.png | Semantic similarity between queries |
| figA2_semantic_scores.png | Semantic recommendation scores |
| figA3_tfidf_vs_semantic.png | TF-IDF vs Semantic comparison |
| figA4_comparison_table.png | Side by side comparison table |

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
```

Run Part 1 (TF-IDF):
```bash
python quran_tfidf_recommendation.py
```

Run Part 2 (Semantic):
```bash
python quran_semantic_search.py
```

## Run Online (No Installation Required)

[![Kaggle Part 1](https://img.shields.io/badge/Kaggle-TF--IDF%20Recommendation-blue)](https://www.kaggle.com/code/shagufakhan/quran-tfidf-recommendation)
[![Kaggle Part 2](https://img.shields.io/badge/Kaggle-Semantic%20Search-blue)](https://www.kaggle.com/code/shagufakhan/quran-semantic-search)

**Part 1 — TF-IDF Recommendation:**
https://www.kaggle.com/code/shagufakhan/quran-tfidf-recommendation

**Part 2 — Semantic Search:**
https://www.kaggle.com/code/shagufakhan/quran-semantic-search

## Key Findings
- Sentence Transformers outperform TF-IDF on indirect expressions
- Semantic search correctly maps metaphorical language to spiritual categories
- All 43 database entries carry explicit evidence levels and hadith references
- System correctly recommends Al-Kahf for Friday and Al-Waqiah for financial difficulty

## Author
Khan Gulrez Shagufa Fazal Ahmed
Independent Researcher, Maharashtra, India

## Citation
If you use this work in your research please cite using the
DOI badge above after Zenodo registration.
