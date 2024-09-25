from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import re
import spacy
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
from word2number import w2n 
# Load NLP models
nlp = spacy.load('en_core_web_sm')
bert_ner_pipeline = pipeline("ner", model="bert-base-uncased", framework="pt")


# Translation pipeline for multilingual support
model_name = "Helsinki-NLP/opus-mt-mul-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
translation_model = MarianMTModel.from_pretrained(model_name)

app = FastAPI()

class RealEstateRequest(BaseModel):
    sentence: str

class RealEstateFeatures(BaseModel):
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    square_feet: Optional[str] = None
    price: Optional[float] = None
    year_built: Optional[str] = None
    address: Optional[str] = None
    radius: Optional[str] = None
    amenities: List[str] = []
    property_types: List[str] = []
    facilities: List[str] = []
    lot_size: Optional[str] = None
    floor_number: Optional[str] = None
    total_floors: Optional[str] = None
    parking: Optional[str] = None
    home_type: Optional[str] = None
    condition: Optional[str] = None
    view: Optional[str] = None
    neighborhood: Optional[str] = None
    is_new_construction: Optional[bool] = None
    nearby_amenities: List[str] = []

    class Config:
        use_enum_values = True
        json_encoders = {
            bool: lambda v: str(v).lower()
        }
def preprocess_text(text):
    # Fix spacing issue for ordinals (e.g., "10 th" -> "10th")
    text = re.sub(r'(\d+)\s+(th|st|nd|rd)', r'\1\2', text)
    return text

def parse_price(price_text: str) -> float:
    # Define suffixes with correct multiplier values
    suffixes = {
        'k': 1_000,
        'K': 1_000,
        'm': 1_000_000,
        'M': 1_000_000,
        'b': 1_000_000_000,
        'B': 1_000_000_000,
        'lakh': 100_000,
        'l': 100_000,
        'crore': 10_000_000,
        'c': 10_000_000
    }
    
    # Extract the numeric part and suffix
    match = re.match(r'(\d+(?:,\d{3})*)([kKmMbBlLcC]?)$', price_text)
    if not match:
        print(f"No match found for price_text: '{price_text}'")  # Debugging statement
        return 0.0
    
    numeric_part = match.group(1).replace(',', '')  # Remove commas
    suffix = match.group(2).lower()  # Normalize suffix to lowercase
    
    # Convert numeric part to float
    price = float(numeric_part)
    
    # Apply suffix multiplier if it exists
    if suffix in suffixes:
        price *= suffixes[suffix]
    
    return price



def translate_text(text: str) -> str:
    # Translate to English if the detected language is not English
    if detect(text) != 'en':
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        translated = translation_model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text
    return text
def text_to_number(text):
    # Convert text numbers to actual numbers
    try:
        return w2n.word_to_num(text)
    except ValueError:
        return None
def is_valid_count(value: str) -> bool:
    try:
        float_value = float(value)
        if(float_value < 10):
            if(float_value % 1 == 0):
                return True
        else:
            return False
        # return 0 < float_value < 10 or float_value % 1 == 0  # Single digit or decimal up to one place
    except ValueError:
        return False
def extract_real_estate_features(text: str) -> RealEstateFeatures:
    text = preprocess_text(text)
    doc = nlp(text)
    bert_results = bert_ner_pipeline(text)
    
    # Initialize the features
    features = RealEstateFeatures()
    address_parts = []
    amenities = set()
    nearby_amenities = set()
    property_types = set()
    facilities = set()
    
    # Keywords
    amenity_keywords = {
        'garage': 'garage', 'kitchen': 'modern kitchen', 'bathroom': 'bathroom',
        'pool': 'swimming pool', 'fitness': 'fitness center', 'fireplace': 'fireplace',
        'yard': 'yard', 'parking': 'parking', 'balcony': 'balcony', 'furnished': 'furnished',
        'living space': 'open-plan living space', 'master bedroom': 'master bedroom',
        'washer': 'washer', 'dryer': 'dryer', 'air conditioning': 'air conditioning',
        'heating': 'heating', 'dishwasher': 'dishwasher', 'microwave': 'microwave',
        'refrigerator': 'refrigerator', 'cable': 'cable TV', 'internet': 'high-speed internet',
        'security system': 'security system', 'pet friendly': 'pet friendly',
        'elevator': 'elevator', 'storage': 'storage space', 'fire pit': 'fire pit',
        'barbecue': 'barbecue area', 'home office': 'home office', 'wet bar': 'wet bar',
        'sauna': 'sauna', 'spa': 'spa', 'greenhouse': 'greenhouse', 'guest house': 'guest house',
        'laundry': 'laundry', 'in-unit laundry': 'in-unit laundry', 'roof top terrace': 'roof top terrace'
    }
    
    property_type_keywords = {
        'apartment': 'apartment', 'condo': 'condo', 'house': 'house', 'studio': 'studio',
        'duplex': 'duplex', 'townhouse': 'townhouse', 'loft': 'loft', 'villa': 'villa',
        'cabin': 'cabin', 'chalet': 'chalet', 'manufactured home': 'manufactured home',
        'mansion': 'mansion', 'farmhouse': 'farmhouse', 'shed': 'shed', 'tiny house': 'tiny house',
        'beach house': 'beach house', 'lake house': 'lake house', 'penthouse': 'penthouse',
        'detached house': 'detached house', 'semi-detached house': 'semi-detached house'
    }
    
    nearby_amenity_keywords = {
        'public transportation': 'public transportation', 'bus stop': 'bus stop',
        'train station': 'train station', 'metro station': 'metro station',
        'local park': 'local park', 'park': 'park', 'grocery store': 'grocery store',
        'supermarket': 'supermarket', 'shopping center': 'shopping center', 'mall': 'mall',
        'restaurant': 'restaurant', 'cafe': 'cafe', 'hospital': 'hospital', 'clinic': 'clinic',
        'school': 'school', 'university': 'university', 'library': 'library', 'gym': 'gym',
        'fitness center': 'fitness center', 'bank': 'bank', 'ATM': 'ATM', 'pharmacy': 'pharmacy',
        'post office': 'post office', 'gas station': 'gas station', 'movie theater': 'movie theater',
        'theater': 'theater', 'museum': 'museum', 'art gallery': 'art gallery',
        'community center': 'community center', 'sports center': 'sports center',
        'swimming pool': 'swimming pool', 'beach': 'beach', 'lake': 'lake', 'river': 'river',
        'bus terminal': 'bus terminal', 'train terminal': 'train terminal'
    }
    # Regex patterns for different elements
    # price_pattern = re.compile(r'\b(?:price|cost|for|budget|total|amount)\s+(\d+(?:,\d{3})*(?:\.\d+)?)([kKmMbB]|lakh|l|crore|c)?\b')
    sqft_pattern = re.compile(r'(\d{1,3}(,\d{3})*)\s*(sq\s*ft|square\s*feet|ft2)', re.IGNORECASE)
    date_pattern = re.compile(r'year\s+(\d{4})', re.IGNORECASE)
    price_pattern = re.compile(
    r'\b(?:price|cost|for|budget|total|amount)?\s*(\d{1,3}(?:,\d{2,3})*|\d+)([kKmMbBlLcC])(?=\s|,)\b'
    )
    # Extract price
    price_match = price_pattern.search(text)
    if price_match:
        numeric_part = price_match.group(1)
        suffix = price_match.group(2) or ''
        price_text = numeric_part + suffix
        print(f"Captured price text: {price_text}")
        features.price = parse_price(price_text)
    
    # Detect nearby amenities
    for keyword, amenity in nearby_amenity_keywords.items():
        if keyword in text:
            nearby_amenities.add(amenity)
    
    # Detect features using Spacy
    for ent in doc.ents:
        word = ent.text.strip()
        label = ent.label_
        print(label,word)
        if label == 'QUANTITY':
            if any(term in word.lower() for term in ['square', 'feet', 'sq', 'ft']):
                features.square_feet = word
            elif 'mile' in word or 'miles' in word:
                features.radius = word
            elif 'acre' in word:
                features.lot_size = word
            elif not features.bedrooms:
                if(is_valid_count(word)):
                    features.bedrooms = word
            elif not features.bathrooms:
                    features.bathrooms = word
        elif label == 'ORDINAL':
            # Handle ordinals like "10th"
            if 'floor' in text:
                features.floor_number = word
        elif label == 'CARDINAL':
            print(word)
            if 'mile' in word or 'miles' in word:
                features.radius = word
            elif 'floor' in word:
                features.floor_number = word
            elif not features.bedrooms:
                if(is_valid_count(word)):
                    features.bedrooms = word
            elif not features.bathrooms:
                    features.bathrooms = word
        elif label == 'MONEY':
            features.price = word
        elif label == 'DATE':
            if re.fullmatch(r'\d{4}', word):
                features.year_built = word
        elif label == 'FAC':
            facilities.add(word)
        elif label in ['GPE', 'LOC', 'FAC', 'ORG']:
            address_parts.append(word)
        elif label in ['ORG', 'PRODUCT'] and 'property' in word:
            features.home_type = word
    
    # Extract amenities and property types using BERT
    for result in bert_results:
        word = result['word'].strip().lower()
        label = result['entity']
        print(label,'::',word)
        if label:
            for keyword, amenity in amenity_keywords.items():
                if keyword in word:
                    amenities.add(amenity)
            for keyword, prop_type in property_type_keywords.items():
                if keyword in word:
                    property_types.add(prop_type)
    
    # Use regex to find square feet if not already found
    if not features.square_feet:
        match = re.search(r'(\d+)\s*(sq\s*ft|square\s*feet|ft2)', text, re.IGNORECASE)
        if match:
            features.square_feet = f"{match.group(1)} {match.group(2)}"
    
    # Assign found values to the feature dictionary
    features.amenities = list(amenities)
    features.property_types = list(property_types)
    features.facilities = list(facilities)
    features.address = ' '.join(address_parts) if address_parts else None
    features.nearby_amenities = list(nearby_amenities)

    return features

@app.post('/parse_input')
def parse_input(request: RealEstateRequest):
    try:
        translated_sentence = translate_text(request.sentence)
        features = extract_real_estate_features(translated_sentence)
        return features
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing the input: {str(e)}")
