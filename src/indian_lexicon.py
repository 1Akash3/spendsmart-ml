"""Indian / UPI domain vocabulary + training-data generation for the categorizer.

Why this exists
---------------
Real UPI statements (GPay/PhonePe/Paytm) render merchants as *space-stripped, often upper-case*
strings built from Indian business-naming conventions:

    SHIVAJISERVICESTATION   MADHURSWEETS   HEALTHFIRSTMEDICO   SWARAJSUPERMARKET

A categorizer trained on Western card merchants ("SQ *COFFEE", "AMZN Mktp") or clean English
("Bought a laptop") never sees this token shape, so its char n-grams cannot fire. Measured on a
real GPay statement, such a model scored 38% on identifiable businesses.

This module supplies the missing signal as *training data*:
  1. `BRANDS`            — real Indian brands per category.
  2. `BUSINESS_WORDS`    — generic Indian business-type suffixes ("ServiceStation", "Caterers",
                           "Medico", "Bazar"). These are the real generalizable signal: any
                           "<person/place>SERVICESTATION" is fuel, whoever runs it.
  3. `NAME_PARTS`        — common Indian name/place prefixes used to build realistic merchant
                           strings, so the model learns "<prefix><business-word>" as a shape
                           rather than memorizing individual merchants.
  4. `upi_variants()`    — reformats any labeled text into UPI shapes (concatenated / upper-case),
                           used to augment the global corpora too.
"""
from __future__ import annotations

import random

# --------------------------------------------------------------------------------------
# 1. Real Indian brands, per project category.
# --------------------------------------------------------------------------------------
BRANDS: dict[str, list[str]] = {
    "food_dining": ["Zomato", "Swiggy", "Dominos Pizza", "Pizza Hut", "McDonalds", "KFC India",
                    "Burger King", "Haldiram", "Bikanervala", "Barbeque Nation", "Cafe Coffee Day",
                    "Starbucks India", "Chaayos", "Wow Momo", "Faasos", "Behrouz Biryani",
                    "Frozen Bottle", "Keventers", "Naturals Ice Cream", "Baskin Robbins",
                    "Subway India", "Theobroma", "Irani Cafe", "Rameshwaram Cafe", "Sagar Ratna"],
    "groceries": ["BigBasket", "Blinkit", "Zepto", "JioMart", "Dunzo", "DMart", "Reliance Fresh",
                  "Reliance Smart", "More Supermarket", "Spencers Retail", "Nature Basket",
                  "Star Bazaar", "Vishal Mega Mart", "Amul Parlour", "Mother Dairy"],
    "transportation": ["Ola Cabs", "Uber India", "Rapido", "RedBus", "IRCTC", "Goibibo",
                       "MakeMyTrip", "Yatra", "Cleartrip", "Ixigo", "Indian Oil", "HP Petrol Pump",
                       "Bharat Petroleum", "Hindustan Petroleum", "Shell India", "Nayara Energy",
                       "FASTag Recharge", "Indigo Airlines", "SpiceJet", "Air India", "Vistara",
                       "PMPML", "BEST Undertaking", "Delhi Metro", "Namma Metro", "Blusmart"],
    "utilities": ["Bharti Airtel", "Jio Recharge", "Vodafone Idea", "BSNL", "Tata Play", "Dish TV",
                  "Hathway Broadband", "ACT Fibernet", "Mahavitaran", "MSEDCL", "BESCOM", "BSES",
                  "Tata Power", "Adani Electricity", "Torrent Power", "Indane Gas", "HP Gas",
                  "Bharat Gas", "Mahanagar Gas", "Water Board"],
    "shopping": ["Zudio", "Myntra", "Ajio", "Amazon India", "Flipkart", "Meesho", "Nykaa",
                 "Snapdeal", "Pantaloons", "Westside", "Max Fashion", "Lifestyle Stores",
                 "Reliance Trends", "Decathlon India", "Croma", "Reliance Digital", "Titan",
                 "Tanishq", "Bata India", "Metro Shoes"],
    "entertainment": ["BookMyShow", "PVR Cinemas", "INOX", "Cinepolis", "Imagica", "Wonderla",
                      "Essel World", "Adlabs", "Smaaash", "Gaming Zone"],
    "subscriptions": ["Netflix India", "Amazon Prime", "Disney Hotstar", "SonyLIV", "ZEE5",
                      "JioCinema", "Spotify India", "YouTube Premium", "Google Play", "Apple Services",
                      "Audible India"],
    "healthcare": ["Apollo Pharmacy", "MedPlus", "Netmeds", "PharmEasy", "Tata 1mg", "Practo",
                   "Wellness Forever", "Fortis Hospital", "Manipal Hospital", "Max Healthcare",
                   "Ruby Hall Clinic", "Dr Lal PathLabs", "Thyrocare", "Metropolis Labs"],
    "financial": ["LIC India", "HDFC Life", "ICICI Prudential", "SBI Life", "Bajaj Finserv",
                  "Groww", "Zerodha", "Upstox", "Policybazaar", "Paytm Money", "NPS Contribution"],
    "housing": ["Society Maintenance", "Nobroker Rent", "Housing Society", "Flat Rent Payment"],
    "misc": ["Wine Shop", "Pan Shop", "Temple Donation", "Xerox Centre", "Courier Service",
             "Blue Dart", "DTDC Courier", "India Post"],
}

# --------------------------------------------------------------------------------------
# 2. Generic Indian business-type words — the real generalizable signal.
# --------------------------------------------------------------------------------------
BUSINESS_WORDS: dict[str, list[str]] = {
    "food_dining": ["Sweets", "Caterers", "Catering", "Cafe", "Restaurant", "Dhaba", "PureVeg",
                    "Thali", "Wadewale", "Juice Centre", "Snacks", "Bakery", "Hotel", "Bhojanalay",
                    "Mess", "Tiffin Service", "Pav Bhaji", "Vada Pav", "Chinese Corner", "Tea Stall",
                    "Chai Point", "Misal House", "Biryani House", "Family Restaurant", "Food Court",
                    "Ice Cream Parlour", "Dosa Corner", "Tandoori", "Fast Food", "Canteen",
                    "Coffee House", "Eatery", "Kitchen", "Rasoi", "Bhavan", "Udupi"],
    "groceries": ["Supermarket", "Super Market", "Bazar", "Bazaar", "Kirana Stores", "General Kirana",
                  "Fruits And Vegetables", "Vegetable Market", "Dairy", "Milk Centre",
                  "Provision Stores", "Grocery Stores", "Rice Traders", "Dal Mill", "Sabzi Mandi",
                  "Fresh Mart", "Daily Needs", "Agro Mart"],
    "transportation": ["Service Station", "Petroleum", "Petrol Pump", "Fuel Centre", "Filling Station",
                       "Auto Services", "Tyres Service", "Tyre Works", "Motor Works", "Garage",
                       "Travels", "Tours And Travels", "Roadways", "Transport Service", "Cab Service",
                       "Auto Rickshaw", "Car Detailing", "Vehicle Service", "Bus Service",
                       "Cable Car", "Toll Plaza", "Parking Services"],
    "healthcare": ["Medical", "Medico", "Medical Stores", "Pharmacy", "Chemist", "Hospital",
                   "Clinic", "Nursing Home", "Diagnostics", "Pathology Lab", "Dental Care",
                   "Eye Care", "Health Care", "Medicos And General", "Ayurvedic Centre",
                   "Physiotherapy", "Drug House"],
    "shopping": ["Stationery", "General Stores", "Stationery And General Stores", "Traders",
                 "Cloth Centre", "Garments", "Fashion Point", "Footwear", "Readymade",
                 "Xerox And Stationary", "Mobile Shop", "Electronics", "Hardware Stores",
                 "Gift Centre", "Toy Shop", "Book Depot", "Novelty Stores"],
    "utilities": ["Electricity Board", "Power Supply", "Gas Agency", "Water Supply",
                  "Broadband Services", "Cable Network", "Mobile Recharge Centre"],
    "entertainment": ["Cinema", "Multiplex", "Talkies", "Water Park", "Amusement Park",
                      "Gaming Zone", "Play Zone", "Club And Resort"],
    "housing": ["Housing Society", "Apartment Maintenance", "Builders And Developers",
                "Property Services", "Rent Collection"],
    "financial": ["Insurance Services", "Finance Company", "Credit Society", "Investment Services",
                  "Loan Services", "Nidhi Limited"],
    "misc": ["Wines", "Wine Shop", "Pan Shop", "Enterprises", "Agencies", "Trading Company",
             "Xerox Centre", "Courier Services", "Salon", "Beauty Parlour", "Tailors",
             "Laundry Services", "Temple Trust", "Charitable Trust", "Photo Studio"],
}

# --------------------------------------------------------------------------------------
# 3. Common Indian name / place prefixes (merchants are usually "<owner or deity or place><type>").
# --------------------------------------------------------------------------------------
NAME_PARTS = [
    "Shivaji", "Sanjay", "Krushna", "Krishna", "Ganesh", "Sai", "Om Sai", "Shree", "Shri",
    "Mauli", "Tulja Bhawani", "Annapurna", "Madhur", "Swaraj", "Prajakta", "Agastya", "Mahendra",
    "Dhanshri", "Aaradhya", "Tirupati", "Guru Krupa", "Gurudatta", "Rajendra", "Vishwaraj",
    "Bhagyashri", "Sharda", "Laxmi", "Vishnu", "Balaji", "Datta", "Siddhi", "Vinayak", "Riddhi",
    "Ashirwad", "Sankalp", "Sanskruti", "Yash", "Rohit", "Amol", "Nitin", "Pooja", "Priya",
    "Sundar", "Anand", "Deepak", "Suresh", "Ramesh", "Mahesh", "Rajesh", "Sagar", "Akash",
    "Pune", "Nashik", "Kolhapur", "Solapur", "Nagpur", "Mumbai", "Thane", "Satara", "Latur",
    "New", "Royal", "Super", "Star", "National", "Modern", "City", "Metro", "Grand", "Classic",
]

_SUFFIX_TAGS = ["", " Pune", " Nagar", " Road", " Chowk", " Branch", " Ltd", " Pvt Ltd", " And Sons"]


def upi_variants(text: str) -> list[str]:
    """UPI-shaped renderings of a description: concatenated, upper-case, and title-concatenated.

    'Madhur Sweets' -> ['MADHURSWEETS', 'MadhurSweets', 'Madhur Sweets']
    This is what teaches char n-grams the space-stripped token shape real statements use.
    """
    t = " ".join(str(text).split())
    if not t:
        return []
    joined = t.replace(" ", "").replace("-", "")
    return [joined.upper(), joined, t]


def generate_upi_training_data(n_per_pattern: int = 6, seed: int = 42) -> list[tuple[str, str]]:
    """Build (description, category) pairs shaped like real UPI merchant strings.

    Combines brands and generic business words with Indian name/place prefixes, then renders each
    in UPI shapes. The generic "<prefix><business-word>" patterns are the point: they generalize to
    unseen merchants that follow the same Indian naming conventions.
    """
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []

    for cat, brands in BRANDS.items():
        for b in brands:
            for v in upi_variants(b):
                rows.append((v, cat))
            for _ in range(2):  # brand + branch/locality tail, as statements often show
                rows.append((rng.choice(upi_variants(b + rng.choice(_SUFFIX_TAGS))), cat))

    for cat, words in BUSINESS_WORDS.items():
        for w in words:
            for v in upi_variants(w):           # bare business word
                rows.append((v, cat))
            for _ in range(n_per_pattern):      # "<prefix> <business word> [tail]"
                name = f"{rng.choice(NAME_PARTS)} {w}{rng.choice(_SUFFIX_TAGS)}"
                rows.append((rng.choice(upi_variants(name)), cat))

    return rows
