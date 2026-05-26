# Cell 1 - Install
!pip install scikit-learn pandas numpy matplotlib seaborn -q

# Cell 2 - Complete Quranic Guidance Recommendation System

"""
AI-Powered Quranic Verse and Surah Recommendation System
=========================================================
Part 1 : Content-Based Recommendation (TF-IDF + Cosine Similarity)
Part 2 : NLP Semantic Search (Natural Language Input)

DISCLAIMER: This system is for spiritual reflection and educational
reference only. It does not constitute religious rulings (fatwas).
All recommendations are based on well-established Quranic themes
and authentic hadith sources cited with evidence levels.

Evidence Levels:
  A = Sahih (authentic) hadith - strongest
  B = Quranic theme - from surah content and classical tafsir
  C = Scholarly consensus
"""

import os, warnings, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

BLUE   = "#1e3a5f"
TEAL   = "#17a2b8"
CORAL  = "#e8533f"
GREEN  = "#28a745"
GOLD   = "#c9a227"
PURPLE = "#7F77DD"
GRAY   = "#6c757d"

plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*65)
print("QURANIC GUIDANCE RECOMMENDATION SYSTEM")
print("="*65)

# ============================================================================
# EMBEDDED DATASET
# ============================================================================

SURAHS = [
    {
        "id": 1, "surah_number": 1,
        "name_arabic": "الفاتحة", "name_english": "Al-Fatihah",
        "total_ayahs": 7,
        "categories": ["healing","illness","all_situations","prayer","anxiety"],
        "themes": "healing guidance mercy opening prayer protection guidance all situations",
        "keywords": "healing illness prayer guidance protection opening every salah sickness",
        "evidence": "A",
        "source": "Sahih Bukhari 5736 - recited for healing",
        "note": "Recited in every unit of prayer; scholars cite it for healing"
    },
    {
        "id": 2, "surah_number": 2,
        "name_arabic": "البقرة", "name_english": "Al-Baqarah",
        "total_ayahs": 286,
        "categories": ["protection","friday","home","night","general"],
        "themes": "protection guidance faith patience legislation home blessings",
        "keywords": "protection home evil eye magic sihr night recitation general guidance",
        "evidence": "A",
        "source": "Sahih Muslim 780 - recitation protects home from Shaytan",
        "note": "Last two verses especially recommended - Al-Baqarah 2:285-286"
    },
    {
        "id": 3, "surah_number": 12,
        "name_arabic": "يوسف", "name_english": "Surah Yusuf",
        "total_ayahs": 111,
        "categories": ["patience","trial","hope","grief","separation"],
        "themes": "patience hope trial hardship family separation trust in Allah",
        "keywords": "patience trial separation grief hope family trust Allah plans",
        "evidence": "B",
        "source": "Quranic theme - story of Prophet Yusuf (AS)",
        "note": "Best of stories - teaches patience through the most severe of trials"
    },
    {
        "id": 4, "surah_number": 18,
        "name_arabic": "الكهف", "name_english": "Al-Kahf",
        "total_ayahs": 110,
        "categories": ["friday","protection","trial","faith"],
        "themes": "friday worship protection from Dajjal trial faith perseverance",
        "keywords": "friday jumu'ah protection Dajjal trial faith light",
        "evidence": "A",
        "source": "Sahih Muslim 809 - recite on Friday for light between two Fridays",
        "note": "First and last 10 verses protect from Dajjal"
    },
    {
        "id": 5, "surah_number": 19,
        "name_arabic": "مريم", "name_english": "Maryam",
        "total_ayahs": 98,
        "categories": ["family","hope","pregnancy","trust","miracle"],
        "themes": "family hope trust miracle children prayer dua answered",
        "keywords": "family children pregnancy birth hope trust miracle dua answered",
        "evidence": "B",
        "source": "Quranic theme - stories of Maryam (AS) and Zakariyya (AS)",
        "note": "Themes of answered prayer and hope for children"
    },
    {
        "id": 6, "surah_number": 20,
        "name_arabic": "طه", "name_english": "Ta-Ha",
        "total_ayahs": 135,
        "categories": ["anxiety","ease","guidance","exams","stress"],
        "themes": "ease difficulty guidance heart patience stress anxiety",
        "keywords": "ease anxiety stress difficulty guidance exam heart expand chest",
        "evidence": "B",
        "source": "Quranic theme - verse 25-26 Dua of Musa (AS) for ease",
        "note": "Rabbish rahli sadri - dua for expanding the chest"
    },
    {
        "id": 7, "surah_number": 55,
        "name_arabic": "الرحمن", "name_english": "Ar-Rahman",
        "total_ayahs": 78,
        "categories": ["gratitude","blessings","thankfulness","reflection"],
        "themes": "gratitude blessings mercy thankfulness reflection appreciation",
        "keywords": "gratitude blessings thankful appreciation mercy bounties reflection",
        "evidence": "B",
        "source": "Quranic theme - repeated verse asking about denying blessings",
        "note": "Fa-bi-ayyi alai rabbikuma tukadhdhibaan - gratitude and reflection"
    },
    {
        "id": 8, "surah_number": 56,
        "name_arabic": "الواقعة", "name_english": "Al-Waqiah",
        "total_ayahs": 96,
        "categories": ["financial_difficulty","rizq","provision","poverty"],
        "themes": "provision rizq wealth poverty financial difficulty blessings",
        "keywords": "provision rizq wealth financial difficulty poverty income sustenance",
        "evidence": "A",
        "source": "Ibn Asakir - recite daily to be free from poverty",
        "note": "Many scholars recommend for sustenance and provision"
    },
    {
        "id": 9, "surah_number": 67,
        "name_arabic": "الملك", "name_english": "Al-Mulk",
        "total_ayahs": 30,
        "categories": ["night","protection","death","afterlife","sleep"],
        "themes": "night protection death afterlife intercession sleep protection grave",
        "keywords": "night sleep protection grave death afterlife intercession daily",
        "evidence": "A",
        "source": "Tirmidhi 2891 - intercedes for reciter in the grave",
        "note": "Recite every night before sleep"
    },
    {
        "id": 10, "surah_number": 73,
        "name_arabic": "المزمل", "name_english": "Al-Muzzammil",
        "total_ayahs": 20,
        "categories": ["tahajjud","night_prayer","patience","strength"],
        "themes": "night prayer tahajjud patience strength worship devotion",
        "keywords": "tahajjud night prayer qiyam strength patience worship devotion",
        "evidence": "B",
        "source": "Quranic theme - command for night prayer and patience",
        "note": "Foundation of tahajjud and night worship"
    },
    {
        "id": 11, "surah_number": 78,
        "name_arabic": "النبأ", "name_english": "An-Naba",
        "total_ayahs": 40,
        "categories": ["reflection","afterlife","accountability","death"],
        "themes": "afterlife accountability reflection death resurrection judgment",
        "keywords": "afterlife death reflection accountability resurrection judgment",
        "evidence": "B",
        "source": "Quranic theme - reflection on creation and accountability",
        "note": "Encourages reflection on Allah's creation and the Day of Judgment"
    },
    {
        "id": 12, "surah_number": 89,
        "name_arabic": "الفجر", "name_english": "Al-Fajr",
        "total_ayahs": 30,
        "categories": ["patience","dawn","soul","peace","contentment"],
        "themes": "patience dawn soul peace contentment tranquility satisfied soul",
        "keywords": "patience dawn soul peace contentment tranquility satisfied",
        "evidence": "B",
        "source": "Quranic theme - nafs al-mutmainnah, satisfied soul",
        "note": "Ends with call to the satisfied soul to enter paradise"
    },
    {
        "id": 13, "surah_number": 93,
        "name_arabic": "الضحى", "name_english": "Ad-Duha",
        "total_ayahs": 11,
        "categories": ["sadness","despair","hope","loneliness","anxiety","depression"],
        "themes": "hope despair sadness loneliness abandoned care mercy love",
        "keywords": "sadness despair hope lonely abandoned forgotten love care grief depression",
        "evidence": "B",
        "source": "Quranic theme - Allah's reassurance to the Prophet (SAW)",
        "note": "Your Lord has not forsaken you - for times of feeling abandoned"
    },
    {
        "id": 14, "surah_number": 94,
        "name_arabic": "الشرح", "name_english": "Ash-Sharh",
        "total_ayahs": 8,
        "categories": ["anxiety","stress","relief","difficulty","hardship"],
        "themes": "relief ease difficulty hardship after ease stress anxiety burden",
        "keywords": "anxiety stress relief ease difficulty hardship indeed with ease",
        "evidence": "B",
        "source": "Quranic theme - with hardship comes ease, twice promised",
        "note": "Indeed with hardship comes ease - promised twice in this surah"
    },
    {
        "id": 15, "surah_number": 103,
        "name_arabic": "العصر", "name_english": "Al-Asr",
        "total_ayahs": 3,
        "categories": ["time","patience","faith","righteous_deeds","reminder"],
        "themes": "time patience faith righteous deeds reminder loss urgency",
        "keywords": "time patience faith deeds reminder loss urgency daily",
        "evidence": "B",
        "source": "Quranic theme - Imam Shafi said this surah alone suffices",
        "note": "Short but comprehensive - reminder about using time wisely"
    },
    {
        "id": 16, "surah_number": 108,
        "name_arabic": "الكوثر", "name_english": "Al-Kawthar",
        "total_ayahs": 3,
        "categories": ["grief","loss","blessing","abundance","hope"],
        "themes": "abundance blessing hope grief loss prayer gratitude",
        "keywords": "grief loss abundance blessing hope prayer gratitude consolation",
        "evidence": "B",
        "source": "Quranic theme - revealed during time of loss and grief",
        "note": "Revealed as consolation - indeed with you is abundance"
    },
    {
        "id": 17, "surah_number": 109,
        "name_arabic": "الكافرون", "name_english": "Al-Kafirun",
        "total_ayahs": 6,
        "categories": ["faith","identity","clarity","night_prayer","doubt"],
        "themes": "faith identity clarity worship devotion night prayer fajr",
        "keywords": "faith identity clarity doubt worship fajr witr night sleep",
        "evidence": "A",
        "source": "Tirmidhi 2899 - recite before sleep and in witr",
        "note": "Declaration of faith - recite before sleep"
    },
    {
        "id": 18, "surah_number": 112,
        "name_arabic": "الإخلاص", "name_english": "Al-Ikhlas",
        "total_ayahs": 4,
        "categories": ["all_situations","tawhid","dhikr","sleep","love"],
        "themes": "tawhid oneness Allah dhikr love worship all situations",
        "keywords": "all situations tawhid oneness dhikr love third Quran sleep morning",
        "evidence": "A",
        "source": "Sahih Bukhari 5013 - equals one third of the Quran",
        "note": "Recite 3x equals reading full Quran in reward"
    },
    {
        "id": 19, "surah_number": 113,
        "name_arabic": "الفلق", "name_english": "Al-Falaq",
        "total_ayahs": 5,
        "categories": ["protection","evil_eye","fear","night","anxiety"],
        "themes": "protection evil eye envy magic fear night anxiety safety",
        "keywords": "protection evil eye envy magic fear night morning safety harm",
        "evidence": "A",
        "source": "Abu Dawud 5082 - recite morning and evening for protection",
        "note": "Recite 3x morning and evening for protection"
    },
    {
        "id": 20, "surah_number": 114,
        "name_arabic": "الناس", "name_english": "An-Nas",
        "total_ayahs": 6,
        "categories": ["protection","anxiety","whispers","night","fear"],
        "themes": "protection whispers anxiety fear inner thoughts night safety",
        "keywords": "protection whispers anxiety fear inner thoughts night morning waswas",
        "evidence": "A",
        "source": "Abu Dawud 5082 - recite morning and evening for protection",
        "note": "Seeking refuge from whispers and inner anxiety"
    },
]

VERSES = [
    {
        "id": 1, "reference": "Al-Baqarah 2:255",
        "name": "Ayat al-Kursi",
        "arabic": "اللَّهُ لَا إِلَٰهَ إِلَّا هُوَ الْحَيُّ الْقَيُّومُ",
        "translation": "Allah - there is no deity except Him, the Ever-Living, the Sustainer of existence.",
        "categories": ["protection","night","sleep","all_situations","fear"],
        "keywords": "protection night sleep fear all situations throne verse powerful",
        "evidence": "A",
        "source": "Sahih Bukhari 2311 - greatest verse in the Quran"
    },
    {
        "id": 2, "reference": "Al-Baqarah 2:286",
        "name": "La Yukallifu Allah",
        "arabic": "لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا",
        "translation": "Allah does not burden a soul beyond that it can bear.",
        "categories": ["anxiety","stress","hardship","trial","overwhelmed"],
        "keywords": "anxiety stress burden capacity overwhelmed hardship trial",
        "evidence": "A",
        "source": "Quran 2:286 - no soul burdened beyond capacity"
    },
    {
        "id": 3, "reference": "Al-Inshirah 94:5-6",
        "name": "Fa-inna ma'al usri yusra",
        "arabic": "فَإِنَّ مَعَ الْعُسْرِ يُسْرًا",
        "translation": "For indeed, with hardship will be ease.",
        "categories": ["anxiety","stress","hardship","difficulty","hope"],
        "keywords": "hardship ease difficulty anxiety stress hope relief promise",
        "evidence": "A",
        "source": "Quran 94:5-6 - ease promised twice with hardship"
    },
    {
        "id": 4, "reference": "Ad-Duha 93:3",
        "name": "Ma wadda'aka rabbuka",
        "arabic": "مَا وَدَّعَكَ رَبُّكَ وَمَا قَلَىٰ",
        "translation": "Your Lord has not forsaken you, nor has He detested you.",
        "categories": ["sadness","loneliness","despair","depression","hope"],
        "keywords": "sadness lonely abandoned forgotten forsaken depression hope love",
        "evidence": "A",
        "source": "Quran 93:3 - revealed during silence of revelation"
    },
    {
        "id": 5, "reference": "Al-Anbiya 21:87",
        "name": "Dua of Yunus (AS)",
        "arabic": "لَّا إِلَٰهَ إِلَّا أَنتَ سُبْحَانَكَ إِنِّي كُنتُ مِنَ الظَّالِمِينَ",
        "translation": "There is no deity except You; exalted are You. Indeed, I have been of the wrongdoers.",
        "categories": ["distress","repentance","trial","difficulty","seeking_forgiveness"],
        "keywords": "distress trial difficulty darkness repentance forgiveness dua accepted",
        "evidence": "A",
        "source": "Tirmidhi 3505 - dua never rejected when recited sincerely"
    },
    {
        "id": 6, "reference": "Ar-Ra'd 13:28",
        "name": "Ala bidhikrillah",
        "arabic": "أَلَا بِذِكْرِ اللَّهِ تَطْمَئِنُّ الْقُلُوبُ",
        "translation": "Verily, in the remembrance of Allah do hearts find rest.",
        "categories": ["anxiety","peace","heart","dhikr","restlessness"],
        "keywords": "peace heart rest anxiety dhikr remembrance tranquility calm",
        "evidence": "A",
        "source": "Quran 13:28 - foundation of dhikr for heart peace"
    },
    {
        "id": 7, "reference": "Al-Imran 3:173",
        "name": "Hasbunallah wa ni'mal wakeel",
        "arabic": "حَسْبُنَا اللَّهُ وَنِعْمَ الْوَكِيلُ",
        "translation": "Sufficient for us is Allah, and He is the best Disposer of affairs.",
        "categories": ["fear","anxiety","trust","trial","reliance"],
        "keywords": "fear anxiety trust reliance tawakkul trial difficult situation",
        "evidence": "A",
        "source": "Sahih Bukhari 4563 - recited by Ibrahim (AS) in the fire"
    },
    {
        "id": 8, "reference": "Al-Baqarah 2:155-157",
        "name": "Inna lillahi wa inna ilayhi raji'un",
        "arabic": "إِنَّا لِلَّهِ وَإِنَّا إِلَيْهِ رَاجِعُونَ",
        "translation": "Indeed, we belong to Allah, and indeed to Him we shall return.",
        "categories": ["death","grief","loss","patience","bereavement"],
        "keywords": "death grief loss bereavement patience inna lillahi return",
        "evidence": "A",
        "source": "Quran 2:155-157 - said upon affliction and loss"
    },
    {
        "id": 9, "reference": "Ta-Ha 20:25-26",
        "name": "Rabbish rahli sadri",
        "arabic": "رَبِّ اشْرَحْ لِي صَدْرِي وَيَسِّرْ لِي أَمْرِي",
        "translation": "My Lord, expand for me my breast and ease for me my task.",
        "categories": ["exams","anxiety","stress","public_speaking","difficulty"],
        "keywords": "exams anxiety stress expand chest ease task public speaking difficulty",
        "evidence": "A",
        "source": "Quran 20:25-26 - Dua of Musa (AS)"
    },
    {
        "id": 10, "reference": "Al-Baqarah 2:201",
        "name": "Rabbana atina fid dunya hasana",
        "arabic": "رَبَّنَا آتِنَا فِي الدُّنْيَا حَسَنَةً وَفِي الْآخِرَةِ حَسَنَةً",
        "translation": "Our Lord, give us good in this world and good in the Hereafter.",
        "categories": ["general","marriage","travel","all_situations","prayer"],
        "keywords": "general all situations marriage travel blessing good dunya akhirah",
        "evidence": "A",
        "source": "Sahih Bukhari 4522 - most complete dua for all situations"
    },
    {
        "id": 11, "reference": "Al-Isra 17:80",
        "name": "Dua for travel and new beginnings",
        "arabic": "رَّبِّ أَدْخِلْنِي مُدْخَلَ صِدْقٍ وَأَخْرِجْنِي مُخْرَجَ صِدْقٍ",
        "translation": "My Lord, cause me to enter a sound entrance and to exit a sound exit.",
        "categories": ["travel","journey","new_beginning","migration"],
        "keywords": "travel journey new beginning entering leaving migration",
        "evidence": "A",
        "source": "Quran 17:80 - dua for entering and exiting"
    },
    {
        "id": 12, "reference": "Yunus 10:57",
        "name": "Healing from what is in the chests",
        "arabic": "وَشِفَاءٌ لِّمَا فِي الصُّدُورِ",
        "translation": "And a healing for what is in the breasts - the Quran itself is healing.",
        "categories": ["illness","healing","pain","mental_health","heart"],
        "keywords": "illness healing pain mental health chest heart Quran cure",
        "evidence": "A",
        "source": "Quran 10:57 - Quran as healing for what is in the chest"
    },
    {
        "id": 13, "reference": "Az-Zumar 39:53",
        "name": "La taqnatu min rahmatillah",
        "arabic": "لَا تَقْنَطُوا مِن رَّحْمَةِ اللَّهِ",
        "translation": "Do not despair of the mercy of Allah.",
        "categories": ["despair","sin","repentance","seeking_forgiveness","hope"],
        "keywords": "despair sin repentance forgiveness hope mercy do not lose hope",
        "evidence": "A",
        "source": "Quran 39:53 - mercy of Allah encompasses all things"
    },
    {
        "id": 14, "reference": "Al-Talaq 65:3",
        "name": "Wa man yatawakkal alallah",
        "arabic": "وَمَن يَتَوَكَّلْ عَلَى اللَّهِ فَهُوَ حَسْبُهُ",
        "translation": "And whoever relies upon Allah - then He is sufficient for him.",
        "categories": ["trust","reliance","anxiety","financial_difficulty","fear"],
        "keywords": "trust reliance tawakkul anxiety financial difficulty fear sufficient",
        "evidence": "A",
        "source": "Quran 65:3 - Allah is sufficient for those who trust in Him"
    },
    {
        "id": 15, "reference": "Al-Baqarah 2:153",
        "name": "Ista'inu bissabri wassalah",
        "arabic": "يَا أَيُّهَا الَّذِينَ آمَنُوا اسْتَعِينُوا بِالصَّبْرِ وَالصَّلَاةِ",
        "translation": "O you who believe, seek help through patience and prayer.",
        "categories": ["anxiety","stress","hardship","patience","prayer"],
        "keywords": "anxiety stress hardship patience prayer help seeking support",
        "evidence": "A",
        "source": "Quran 2:153 - seek help through sabr and salah"
    },
]

DUAS = [
    {
        "id": 1, "name": "Dua for anxiety and grief",
        "arabic": "اللَّهُمَّ إِنِّي أَعُوذُ بِكَ مِنَ الْهَمِّ وَالْحَزَنِ",
        "transliteration": "Allahumma inni a'udhu bika minal-hammi wal-hazan",
        "translation": "O Allah, I seek refuge in You from worry and grief.",
        "categories": ["anxiety","sadness","stress","grief","depression"],
        "keywords": "anxiety worry grief sadness stress depression mental health",
        "evidence": "A",
        "source": "Sahih Bukhari 6369"
    },
    {
        "id": 2, "name": "Dua for illness and healing",
        "arabic": "اللَّهُمَّ رَبَّ النَّاسِ أَذْهِبِ الْبَأْسَ وَاشْفِ",
        "transliteration": "Allahumma Rabb an-nas, adhhib al-ba's washfi",
        "translation": "O Allah, Lord of mankind, remove the harm and heal.",
        "categories": ["illness","pain","healing","recovery","hospital"],
        "keywords": "illness pain healing recovery hospital sick cure health",
        "evidence": "A",
        "source": "Sahih Bukhari 5675 - dua for the sick"
    },
    {
        "id": 3, "name": "Dua for entering and leaving home",
        "arabic": "بِسْمِ اللَّهِ تَوَكَّلْتُ عَلَى اللَّهِ",
        "transliteration": "Bismillahi tawakkaltu alallah",
        "translation": "In the name of Allah, I trust in Allah.",
        "categories": ["travel","daily","protection","morning","general"],
        "keywords": "travel daily protection morning home entering leaving trust",
        "evidence": "A",
        "source": "Abu Dawud 5095"
    },
    {
        "id": 4, "name": "Salawat - Blessings on the Prophet",
        "arabic": "اللَّهُمَّ صَلِّ عَلَى مُحَمَّدٍ",
        "transliteration": "Allahumma salli ala Muhammad",
        "translation": "O Allah, send blessings upon Muhammad (SAW).",
        "categories": ["friday","dhikr","general","daily"],
        "keywords": "friday salawat blessings prophet Muhammad daily dhikr",
        "evidence": "A",
        "source": "Sahih Muslim 408 - abundant salawat on Friday"
    },
    {
        "id": 5, "name": "Dua for financial hardship",
        "arabic": "اللَّهُمَّ اكْفِنِي بِحَلَالِكَ عَنْ حَرَامِكَ",
        "transliteration": "Allahumma-kfini bihalaalika an haraamik",
        "translation": "O Allah, suffice me with what is lawful against what is unlawful.",
        "categories": ["financial_difficulty","provision","rizq","poverty","need"],
        "keywords": "financial difficulty poverty provision rizq lawful need debt",
        "evidence": "A",
        "source": "Tirmidhi 3563"
    },
    {
        "id": 6, "name": "Dua for knowledge and exams",
        "arabic": "رَبِّ زِدْنِي عِلْمًا",
        "transliteration": "Rabbi zidni ilma",
        "translation": "My Lord, increase me in knowledge.",
        "categories": ["exams","knowledge","guidance","learning"],
        "keywords": "exams knowledge guidance learning study increase wisdom",
        "evidence": "A",
        "source": "Quran 20:114"
    },
    {
        "id": 7, "name": "Dua before sleep",
        "arabic": "اللَّهُمَّ بِاسْمِكَ أَمُوتُ وَأَحْيَا",
        "transliteration": "Allahumma bismika amutu wa ahya",
        "translation": "O Allah, in Your name I die and I live.",
        "categories": ["sleep","night","protection","death","daily"],
        "keywords": "sleep night protection death peace rest daily bedtime",
        "evidence": "A",
        "source": "Sahih Bukhari 6312"
    },
    {
        "id": 8, "name": "Istikhara - Seeking Allah's guidance",
        "arabic": "اللَّهُمَّ إِنِّي أَسْتَخِيرُكَ بِعِلْمِكَ",
        "transliteration": "Allahumma inni astakhiruka bi'ilmik",
        "translation": "O Allah, I seek Your guidance through Your knowledge.",
        "categories": ["marriage","decision","guidance","travel","new_beginning"],
        "keywords": "marriage decision guidance travel new beginning choice istikhara",
        "evidence": "A",
        "source": "Sahih Bukhari 1162"
    },
]

CATEGORIES_META = {
    "anxiety":             {"label": "Anxiety & Stress",      "color": CORAL},
    "sadness":             {"label": "Sadness & Grief",        "color": BLUE},
    "illness":             {"label": "Illness & Pain",         "color": TEAL},
    "friday":              {"label": "Friday Worship",         "color": GREEN},
    "protection":          {"label": "Protection",             "color": GOLD},
    "patience":            {"label": "Patience & Trial",       "color": PURPLE},
    "gratitude":           {"label": "Gratitude",              "color": GREEN},
    "seeking_forgiveness": {"label": "Seeking Forgiveness",    "color": CORAL},
    "financial_difficulty":{"label": "Financial Difficulty",   "color": BLUE},
    "night":               {"label": "Night Worship & Sleep",  "color": PURPLE},
    "travel":              {"label": "Travel",                 "color": TEAL},
    "marriage":            {"label": "Marriage & Family",      "color": GOLD},
    "exams":               {"label": "Exams & Knowledge",      "color": GREEN},
    "death":               {"label": "Death & Bereavement",    "color": GRAY},
    "hope":                {"label": "Hope & Relief",          "color": GREEN},
    "repentance":          {"label": "Repentance",             "color": CORAL},
    "general":             {"label": "General Guidance",       "color": BLUE},
    "tahajjud":            {"label": "Tahajjud & Dhikr",       "color": PURPLE},
    "ramadan":             {"label": "Ramadan",                "color": GREEN},
    "dhikr":               {"label": "Dhikr & Remembrance",    "color": TEAL},
}

df_surahs = pd.DataFrame(SURAHS)
df_verses = pd.DataFrame(VERSES)
df_duas   = pd.DataFrame(DUAS)

print(f"\nDataset loaded:")
print(f"  Surahs  : {len(df_surahs)}")
print(f"  Verses  : {len(df_verses)}")
print(f"  Duas    : {len(df_duas)}")

# ============================================================================
# EDA VISUALIZATIONS
# ============================================================================
print("\nGenerating figures...")

# Figure 1: Category coverage
all_cats = []
for s in SURAHS: all_cats.extend(s["categories"])
for v in VERSES: all_cats.extend(v["categories"])
for d in DUAS:   all_cats.extend(d["categories"])

cat_counts = pd.Series(all_cats).value_counts().head(18)
fig, ax = plt.subplots(figsize=(14, 7))
colors_bar = [CATEGORIES_META.get(c, {}).get("color", GRAY)
              for c in cat_counts.index]
bars = ax.barh(cat_counts.index, cat_counts.values,
               color=colors_bar, alpha=0.85)
for bar, val in zip(bars, cat_counts.values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=9, fontweight="bold")
ax.set_title("Quranic Guidance System - Category Coverage",
             fontsize=14, fontweight="bold", color=BLUE, pad=15)
ax.set_xlabel("Number of Entries (Surahs + Verses + Duas)", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_category_coverage.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure 1 saved: Category coverage")

# Figure 2: Evidence quality
all_evidence = ([s["evidence"] for s in SURAHS] +
                [v["evidence"] for v in VERSES] +
                [d["evidence"] for d in DUAS])
ev_labels = {"A": "Sahih Hadith\n(Strongest)",
             "B": "Quranic Theme\n(Tafsir-based)",
             "C": "Scholarly Consensus"}
ev_counts = pd.Series(all_evidence).value_counts()
ev_colors = [GREEN, TEAL, GOLD]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Dataset Quality and Evidence Levels",
             fontsize=14, fontweight="bold", color=BLUE)
axes[0].pie([ev_counts.get(k, 0) for k in ["A","B","C"]],
            labels=[ev_labels.get(k, k) for k in ["A","B","C"]],
            colors=ev_colors, autopct="%1.1f%%", startangle=90,
            pctdistance=0.75)
axes[0].set_title("Evidence Level Distribution",
                  fontweight="bold", color=BLUE)
content_counts = [len(SURAHS), len(VERSES), len(DUAS)]
content_labels = ["Surahs", "Key Verses", "Duas"]
axes[1].bar(content_labels, content_counts,
            color=[BLUE, CORAL, GREEN], alpha=0.85, width=0.5)
for i, v in enumerate(content_counts):
    axes[1].text(i, v + 0.3, str(v), ha="center",
                 fontsize=11, fontweight="bold")
axes[1].set_title("Content Distribution", fontweight="bold", color=BLUE)
axes[1].spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_evidence_quality.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure 2 saved: Evidence quality")

# Figure 3: Category overlap heatmap
main_cats = ["anxiety","sadness","illness","protection","patience",
             "gratitude","financial_difficulty","night","friday","hope"]

def shared_entries(c1, c2):
    count = 0
    for items in [SURAHS, VERSES, DUAS]:
        for item in items:
            if c1 in item["categories"] and c2 in item["categories"]:
                count += 1
    return count

sim_matrix = np.zeros((len(main_cats), len(main_cats)))
for i, c1 in enumerate(main_cats):
    for j, c2 in enumerate(main_cats):
        sim_matrix[i][j] = shared_entries(c1, c2)

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(sim_matrix.astype(int), annot=True, fmt="d",
            cmap="Blues", xticklabels=main_cats,
            yticklabels=main_cats, ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 10})
ax.set_title("Category Overlap Matrix - Shared Guidance Entries",
             fontsize=13, fontweight="bold", color=BLUE, pad=15)
ax.tick_params(axis="x", rotation=30, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_category_overlap.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure 3 saved: Category overlap heatmap")

# ============================================================================
# PART 1 - TF-IDF CONTENT-BASED RECOMMENDATION
# ============================================================================
print("\n" + "="*65)
print("PART 1 - TF-IDF CONTENT-BASED RECOMMENDATION")
print("="*65)

def build_corpus():
    corpus, meta = [], []
    for s in SURAHS:
        text = (s["themes"] + " " + s["keywords"] + " " +
                " ".join(s["categories"]))
        corpus.append(text)
        meta.append({"type":"surah","data":s})
    for v in VERSES:
        text = (v["keywords"] + " " + v["translation"] + " " +
                " ".join(v["categories"]))
        corpus.append(text)
        meta.append({"type":"verse","data":v})
    for d in DUAS:
        text = (d["keywords"] + " " + d["translation"] + " " +
                " ".join(d["categories"]))
        corpus.append(text)
        meta.append({"type":"dua","data":d})
    return corpus, meta

corpus, meta = build_corpus()

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), min_df=1,
    max_features=500, stop_words="english"
)
tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

CATEGORY_QUERIES = {
    "anxiety":             "anxiety stress worry fear nervous overwhelmed burden relief ease",
    "sadness":             "sadness grief sorrow despair depression lonely abandoned hope",
    "illness":             "illness pain sick healing recovery hospital cure health suffering",
    "friday":              "friday jumu'ah worship recitation light week blessed day",
    "protection":          "protection evil eye magic harm enemy danger safe safety refuge",
    "patience":            "patience trial hardship sabr perseverance endurance strength",
    "gratitude":           "gratitude thankful blessings appreciate bounty alhamdulillah",
    "seeking_forgiveness": "forgiveness tawbah repentance sin guilt regret mercy Allah",
    "financial_difficulty":"money poverty debt financial difficulty provision rizq sustenance",
    "night":               "night sleep protection tahajjud qiyam dhikr rest peace",
    "travel":              "travel journey safety protection road trip leaving arrival",
    "marriage":            "marriage nikah spouse family children love relationship",
    "exams":               "exams study knowledge test results nervous education",
    "death":               "death loss bereavement inna lillahi grief mourning patience",
    "hope":                "hope relief ease after hardship promise light future",
    "repentance":          "repentance sin forgiveness guilt turning back Allah mercy",
    "general":             "general guidance all situations daily life blessing",
    "tahajjud":            "tahajjud night prayer qiyam worship devotion ramadan",
    "dhikr":               "dhikr remembrance heart peace tranquility subhanallah",
    "ramadan":             "ramadan fasting iftar suhur laylatul qadr tarawih eid",
}

def recommend(query_text, top_n_surahs=3, top_n_verses=3, top_n_duas=2):
    query_vec = vectorizer.transform([query_text])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    surah_results = [(scores[i], meta[i]["data"])
                     for i in range(len(meta))
                     if meta[i]["type"] == "surah"]
    verse_results = [(scores[i], meta[i]["data"])
                     for i in range(len(meta))
                     if meta[i]["type"] == "verse"]
    dua_results   = [(scores[i], meta[i]["data"])
                     for i in range(len(meta))
                     if meta[i]["type"] == "dua"]
    surah_results.sort(key=lambda x: x[0], reverse=True)
    verse_results.sort(key=lambda x: x[0], reverse=True)
    dua_results.sort(key=lambda x: x[0], reverse=True)
    return {
        "surahs": surah_results[:top_n_surahs],
        "verses": verse_results[:top_n_verses],
        "duas":   dua_results[:top_n_duas]
    }

def format_results(situation, results):
    print(f"\n{'='*65}")
    print(f"GUIDANCE FOR: {situation.upper()}")
    print(f"{'='*65}")
    print("\nRECOMMENDED SURAHS:")
    for score, item in results["surahs"]:
        print(f"  [{item['evidence']}] {item['name_arabic']} - "
              f"{item['name_english']} (score: {score:.3f})")
        print(f"       Source: {item['source']}")
        if item.get("note"):
            print(f"       Note  : {item['note']}")
    print("\nKEY VERSES:")
    for score, item in results["verses"]:
        print(f"  [{item['evidence']}] {item['reference']} - {item['name']}")
        print(f"       {item['arabic']}")
        print(f"       {item['translation']}")
        print(f"       Source: {item['source']}")
    print("\nRELEVANT DUAS:")
    for score, item in results["duas"]:
        print(f"  [{item['evidence']}] {item['name']}")
        print(f"       {item['arabic']}")
        print(f"       {item['translation']}")
        print(f"       Source: {item['source']}")

test_situations = [
    ("Feeling anxious and stressed",      CATEGORY_QUERIES["anxiety"]),
    ("Feeling sad and grieving a loss",   CATEGORY_QUERIES["sadness"]),
    ("Sick and in pain",                  CATEGORY_QUERIES["illness"]),
    ("Friday worship",                    CATEGORY_QUERIES["friday"]),
    ("Financial difficulty",              CATEGORY_QUERIES["financial_difficulty"]),
]

all_results = {}
for situation, query in test_situations:
    results = recommend(query)
    format_results(situation, results)
    all_results[situation] = results

# Figure 4: Recommendation scores
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("TF-IDF Recommendation Scores - Anxiety Query",
             fontsize=14, fontweight="bold", color=BLUE)
first_results = recommend(CATEGORY_QUERIES["anxiety"])
types_data = [
    ("Surahs", first_results["surahs"], BLUE),
    ("Verses", first_results["verses"], CORAL),
    ("Duas",   first_results["duas"],   GREEN),
]
for ax, (label, items, color) in zip(axes, types_data):
    names  = [item[1].get("name_english") or item[1].get("name") or
              item[1].get("reference","")[:20] for item in items]
    scores = [item[0] for item in items]
    ax.barh(names, scores, color=color, alpha=0.85)
    for i, (name, score) in enumerate(zip(names, scores)):
        ax.text(score + 0.002, i, f"{score:.3f}",
                va="center", fontsize=9, fontweight="bold")
    ax.set_title(label, fontweight="bold", color=BLUE, fontsize=12)
    ax.set_xlabel("Cosine Similarity Score")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(0, max(scores) * 1.2 + 0.01 if scores else 1)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_recommendation_scores.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("\nFigure 4 saved: Recommendation scores")

# Figure 5: Full recommendation dashboard
fig = plt.figure(figsize=(18, 14))
gs  = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)
situations_list = list(all_results.keys())
queries_list    = [q for _, q in test_situations]

for idx, (situation, query) in enumerate(
    zip(situations_list[:6], queries_list[:6])):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs[row, col])
    res = recommend(query)
    all_items = (
        [(s[0], s[1].get("name_english",""), "S", BLUE)
         for s in res["surahs"]] +
        [(v[0], v[1].get("reference","")[:18], "V", CORAL)
         for v in res["verses"]] +
        [(d[0], d[1].get("name","")[:18], "D", GREEN)
         for d in res["duas"]]
    )
    all_items.sort(key=lambda x: x[0], reverse=True)
    names  = [f"[{t}] {n[:16]}" for _, n, t, _ in all_items]
    scores = [s for s, _, _, _ in all_items]
    colors = [c for _, _, _, c in all_items]
    ax.barh(names, scores, color=colors, alpha=0.8)
    short = situation[:25] + "..." if len(situation) > 25 else situation
    ax.set_title(short, fontweight="bold", color=BLUE, fontsize=9)
    ax.set_xlabel("Score", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Quranic Guidance Recommendation Dashboard\n"
             "[S]=Surah  [V]=Verse  [D]=Dua",
             fontsize=14, fontweight="bold", color=BLUE, y=1.01)
plt.savefig(f"{OUTPUT_DIR}/fig5_full_dashboard.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure 5 saved: Full recommendation dashboard")

# ============================================================================
# PART 2 - NLP SEMANTIC SEARCH (NATURAL LANGUAGE INPUT)
# ============================================================================
print("\n" + "="*65)
print("PART 2 - NLP SEMANTIC SEARCH")
print("="*65)

EMOTION_LEXICON = {
    "anxiety":   ["anxious","anxiety","worried","worry","nervous","stress",
                  "stressed","panic","scared","overwhelmed","fear","afraid",
                  "restless","uneasy","tense","dread","apprehensive"],
    "sadness":   ["sad","sadness","depressed","depression","crying","lonely",
                  "alone","hopeless","helpless","miserable","grief","grieving",
                  "heartbroken","devastated","abandoned","lost","empty"],
    "illness":   ["sick","ill","illness","pain","hurt","disease","hospital",
                  "treatment","unwell","suffering","ache","fever","recovery"],
    "friday":    ["friday","jumu'ah","jumuah","week","congregation","mosque"],
    "protection":["protect","protection","evil","harm","enemy","danger","safe",
                  "afraid","jinns","magic","envy","evil eye"],
    "patience":  ["patient","patience","waiting","trial","difficult","hard",
                  "struggle","test","tribulation","endure","persevere"],
    "gratitude": ["grateful","gratitude","thankful","appreciate","blessed",
                  "thank","alhamdulillah","blessings","bounty"],
    "seeking_forgiveness":["forgive","forgiveness","sin","guilty","regret",
                  "mistake","wrong","repent","repentance","tawbah","ashamed"],
    "financial_difficulty":["money","poor","poverty","debt","broke","afford",
                  "income","provision","rizq","financial","wealth"],
    "night":     ["night","sleep","insomnia","dark","bedtime","tahajjud",
                  "qiyam","dream","nightmare","rest","peace"],
    "travel":    ["travel","journey","trip","road","flight","leaving","arrival",
                  "departure","visiting","going","moving","migration"],
    "marriage":  ["marriage","married","spouse","husband","wife","nikah",
                  "family","divorce","relationship","love","children","baby"],
    "exams":     ["exam","test","study","learning","knowledge","school",
                  "university","results","grades","nervous","education"],
    "death":     ["death","died","funeral","dead","loss","bereavement",
                  "mourning","passed away","inna lillahi","grave"],
    "hope":      ["hope","hopeful","better","improve","future","relief",
                  "easier","ease","after","positive","optimistic"],
    "ramadan":   ["ramadan","fasting","iftar","suhur","laylatul qadr",
                  "tarawih","eid","fast"],
    "tahajjud":  ["tahajjud","qiyam","night prayer","witr","midnight"],
    "dhikr":     ["dhikr","remember","remembrance","subhanallah",
                  "alhamdulillah","allahu akbar","la ilaha"],
}

def extract_categories_nlp(user_text):
    text_lower = user_text.lower()
    scores = {}
    for cat, keywords in EMOTION_LEXICON.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score
    scores_sorted = sorted(scores.items(),
                           key=lambda x: x[1], reverse=True)
    top_cats = [cat for cat, _ in scores_sorted[:3]]
    if not top_cats:
        top_cats = ["general"]
    return top_cats, scores

def nlp_recommend(user_text):
    top_cats, cat_scores = extract_categories_nlp(user_text)
    combined_query = user_text.lower() + " " + " ".join([
        CATEGORY_QUERIES.get(cat, cat) for cat in top_cats
    ])
    results = recommend(combined_query)
    return results, top_cats, cat_scores

nlp_tests = [
    "I feel so worried and anxious about my future",
    "I am sick and in pain and need healing",
    "I feel very lonely and sad today",
    "I have financial problems and struggling with money",
    "I want to pray Tahajjud and get closer to Allah",
    "It is Friday and I want to know what to recite",
    "I made a mistake and need forgiveness",
    "I am traveling tomorrow and need protection",
]

print("\nNLP RECOMMENDATION RESULTS:")
nlp_results_store = []
for text in nlp_tests:
    results, cats, scores = nlp_recommend(text)
    print(f"\nInput     : '{text}'")
    print(f"Detected  : {cats}")
    nlp_results_store.append({
        "input": text,
        "categories": cats,
        "top_surah": results["surahs"][0][1]["name_english"]
                     if results["surahs"] else "N/A",
        "top_verse": results["verses"][0][1]["reference"]
                     if results["verses"] else "N/A",
        "confidence": round(results["surahs"][0][0], 3)
                      if results["surahs"] else 0
    })
    format_results(text, results)

# Figure 6: NLP category detection
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Part 2 - NLP Category Detection from Natural Language",
             fontsize=14, fontweight="bold", color=BLUE, y=0.98)
for ax, text in zip(axes.flatten(), nlp_tests):
    _, cats, score_dict = nlp_recommend(text)
    if score_dict:
        sorted_scores = sorted(score_dict.items(),
                               key=lambda x: x[1], reverse=True)[:5]
        cat_names = [c for c, _ in sorted_scores]
        cat_vals  = [v for _, v in sorted_scores]
        ax.barh(cat_names, cat_vals, color=CORAL, alpha=0.8)
        ax.set_title(text[:32] + "...", fontsize=8,
                     fontweight="bold", color=BLUE)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Keyword matches", fontsize=8)
    else:
        ax.text(0.5, 0.5, "General guidance",
                ha="center", va="center", fontsize=10, color=GRAY)
        ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_nlp_detection.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("\nFigure 6 saved: NLP category detection")

# Figure 7: NLP summary table
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")
table_data = [[r["input"][:40]+"...", ", ".join(r["categories"]),
               r["top_surah"], r["top_verse"],
               f"{r['confidence']:.3f}"]
              for r in nlp_results_store]
cols = ["User Input", "Detected Categories",
        "Top Surah", "Top Verse", "Confidence"]
table = ax.table(cellText=table_data, colLabels=cols,
                 loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2.2)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor(BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F0F4F8")
    cell.set_edgecolor("white")
ax.set_title("NLP Recommendation Summary Table",
             fontsize=14, fontweight="bold", color=BLUE, pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_nlp_summary_table.png",
            bbox_inches="tight", facecolor="white")
plt.show()
print("Figure 7 saved: NLP summary table")

# ============================================================================
# INTERACTIVE PREDICTION FUNCTION
# ============================================================================
print("\n" + "="*65)
print("SAMPLE INTERACTIVE PREDICTIONS")
print("="*65)

def full_guidance(user_input):
    print(f"\n{'*'*65}")
    print(f"Seeking guidance for: {user_input}")
    print("DISCLAIMER: For spiritual reflection only. Not a fatwa.")
    print(f"{'*'*65}")
    results, cats, _ = nlp_recommend(user_input)
    print(f"Detected situation  : {', '.join(cats)}")
    print("\n--- RECOMMENDED SURAHS ---")
    for score, item in results["surahs"]:
        print(f"\n  {item['name_arabic']} ({item['name_english']})")
        print(f"  Surah {item['surah_number']} | {item['total_ayahs']} verses")
        print(f"  Evidence [{item['evidence']}]: {item['source']}")
        print(f"  Note: {item.get('note','')}")
    print("\n--- KEY VERSES ---")
    for score, item in results["verses"]:
        print(f"\n  {item['reference']}")
        print(f"  {item['arabic']}")
        print(f"  {item['translation']}")
        print(f"  Evidence [{item['evidence']}]: {item['source']}")
    print("\n--- RELEVANT DUAS ---")
    for score, item in results["duas"]:
        print(f"\n  {item['name']}")
        print(f"  {item['arabic']}")
        print(f"  {item['transliteration']}")
        print(f"  {item['translation']}")
        print(f"  Evidence [{item['evidence']}]: {item['source']}")

full_guidance("I am feeling very anxious and cannot sleep at night")
full_guidance("I have an important exam tomorrow and I am nervous")
full_guidance("I lost someone dear to me and I am grieving")

# Save system data
with open(f"{OUTPUT_DIR}/quran_system_data.json", "w",
          encoding="utf-8") as f:
    json.dump({
        "surahs": SURAHS, "verses": VERSES, "duas": DUAS,
        "categories": list(CATEGORIES_META.keys())
    }, f, ensure_ascii=False, indent=2)

print("\n" + "="*65)
print("SYSTEM COMPLETED SUCCESSFULLY")
print("="*65)
print(f"\nDatabase  : {len(SURAHS)} surahs + {len(VERSES)} verses + "
      f"{len(DUAS)} duas = {len(SURAHS)+len(VERSES)+len(DUAS)} total entries")
print(f"Categories: {len(CATEGORIES_META)}")
print(f"Figures   : 7 saved to {OUTPUT_DIR}/")
print("\nNOTE: For enhanced NLP with semantic embeddings on Kaggle:")
print("  !pip install sentence-transformers -q")
print("  from sentence_transformers import SentenceTransformer")
print("  model = SentenceTransformer('all-MiniLM-L6-v2')")
