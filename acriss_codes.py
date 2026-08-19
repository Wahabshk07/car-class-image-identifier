"""ACRISS Category / Type definitions and Stanford Cars → ACRISS mapping."""

from __future__ import annotations

# Category: size / market tier
CATEGORIES: dict[str, str] = {
    "M": "Mini",
    "N": "Mini Elite",
    "E": "Economy",
    "H": "Economy Elite",
    "C": "Compact",
    "D": "Compact Elite",
    "I": "Intermediate",
    "J": "Intermediate Elite",
    "S": "Standard",
    "R": "Standard Elite",
    "F": "Fullsize",
    "G": "Fullsize Elite",
    "P": "Premium",
    "U": "Premium Elite",
    "L": "Luxury",
    "W": "Luxury Elite",
    "O": "Oversize",
    "X": "Special",
}

# Type: body style
TYPES: dict[str, str] = {
    "B": "2-3 Door",
    "C": "2/4 Door",
    "D": "4-5 Door",
    "W": "Wagon/Estate",
    "V": "Passenger Van",
    "L": "Limousine/Sedan",
    "S": "Sport",
    "T": "Convertible",
    "F": "SUV",
    "J": "Open Air All Terrain",
    "X": "Special",
    "P": "Pick up (single/extended cab) 2 door",
    "Q": "Pick up (double cab) 4 door",
    "Z": "Special Offer Car",
    "E": "Coupe",
    "M": "Monospace",
    "R": "Recreational Vehicle",
    "H": "Motor Home",
    "Y": "2 Wheel Vehicle",
    "N": "Roadster",
    "G": "Crossover",
    "K": "Commercial Van/Truck",
}


def folder_name(code: str, label: str) -> str:
    """Safe folder name like M_Mini or F_SUV."""
    safe = label.replace("/", "-").replace(" ", "_")
    return f"{code}_{safe}"


# Stanford Cars class index (1-based, matching cars_meta.mat) → (category, type)
# Mapping is heuristic from make/model/body style in the class name.
STANFORD_TO_ACRISS: dict[int, tuple[str, str]] = {
    1: ("O", "F"),   # AM General Hummer SUV 2000
    2: ("P", "D"),   # Acura RL Sedan 2012
    3: ("P", "D"),   # Acura TL Sedan 2012
    4: ("U", "D"),   # Acura TL Type-S 2008
    5: ("P", "D"),   # Acura TSX Sedan 2012
    6: ("D", "E"),   # Acura Integra Type R 2001
    7: ("P", "G"),   # Acura ZDX Hatchback 2012
    8: ("X", "T"),   # Aston Martin V8 Vantage Convertible 2012
    9: ("X", "E"),   # Aston Martin V8 Vantage Coupe 2012
    10: ("X", "T"),  # Aston Martin Virage Convertible 2012
    11: ("X", "E"),  # Aston Martin Virage Coupe 2012
    12: ("U", "T"),  # Audi RS 4 Convertible 2008
    13: ("P", "E"),  # Audi A5 Coupe 2012
    14: ("U", "E"),  # Audi TTS Coupe 2012
    15: ("X", "E"),  # Audi R8 Coupe 2012
    16: ("P", "D"),  # Audi V8 Sedan 1994
    17: ("P", "D"),  # Audi 100 Sedan 1994
    18: ("P", "W"),  # Audi 100 Wagon 1994
    19: ("P", "D"),  # Audi TT Hatchback 2011
    20: ("U", "D"),  # Audi S6 Sedan 2011
    21: ("U", "T"),  # Audi S5 Convertible 2012
    22: ("U", "E"),  # Audi S5 Coupe 2012
    23: ("U", "D"),  # Audi S4 Sedan 2012
    24: ("U", "D"),  # Audi S4 Sedan 2007
    25: ("U", "E"),  # Audi TT RS Coupe 2012
    26: ("P", "D"),  # BMW ActiveHybrid 5 Sedan 2012
    27: ("P", "T"),  # BMW 1 Series Convertible 2012
    28: ("P", "E"),  # BMW 1 Series Coupe 2012
    29: ("P", "D"),  # BMW 3 Series Sedan 2012
    30: ("P", "W"),  # BMW 3 Series Wagon 2012
    31: ("L", "T"),  # BMW 6 Series Convertible 2007
    32: ("P", "F"),  # BMW X5 SUV 2007
    33: ("U", "F"),  # BMW X6 SUV 2012
    34: ("U", "E"),  # BMW M3 Coupe 2012
    35: ("U", "D"),  # BMW M5 Sedan 2010
    36: ("U", "T"),  # BMW M6 Convertible 2010
    37: ("P", "F"),  # BMW X3 SUV 2012
    38: ("U", "T"),  # BMW Z4 Convertible 2012
    39: ("W", "T"),  # Bentley Continental Supersports Conv. Convertible 2012
    40: ("L", "D"),  # Bentley Arnage Sedan 2009
    41: ("W", "D"),  # Bentley Mulsanne Sedan 2011
    42: ("W", "E"),  # Bentley Continental GT Coupe 2012
    43: ("W", "E"),  # Bentley Continental GT Coupe 2007
    44: ("L", "D"),  # Bentley Continental Flying Spur Sedan 2007
    45: ("X", "T"),  # Bugatti Veyron 16.4 Convertible 2009
    46: ("X", "E"),  # Bugatti Veyron 16.4 Coupe 2009
    47: ("I", "D"),  # Buick Regal GS 2012
    48: ("S", "F"),  # Buick Rainier SUV 2007
    49: ("C", "D"),  # Buick Verano Sedan 2012
    50: ("S", "F"),  # Buick Enclave SUV 2012
    51: ("U", "D"),  # Cadillac CTS-V Sedan 2012
    52: ("P", "F"),  # Cadillac SRX SUV 2012
    53: ("O", "Q"),  # Cadillac Escalade EXT Crew Cab 2007
    54: ("O", "Q"),  # Chevrolet Silverado 1500 Hybrid Crew Cab 2012
    55: ("R", "T"),  # Chevrolet Corvette Convertible 2012
    56: ("R", "E"),  # Chevrolet Corvette ZR1 2012
    57: ("R", "E"),  # Chevrolet Corvette Ron Fellows Edition Z06 2007
    58: ("S", "F"),  # Chevrolet Traverse SUV 2012
    59: ("R", "T"),  # Chevrolet Camaro Convertible 2012
    60: ("C", "D"),  # Chevrolet HHR SS 2010
    61: ("F", "D"),  # Chevrolet Impala Sedan 2007
    62: ("F", "F"),  # Chevrolet Tahoe Hybrid SUV 2012
    63: ("E", "D"),  # Chevrolet Sonic Sedan 2012
    64: ("O", "K"),  # Chevrolet Express Cargo Van 2007
    65: ("O", "Q"),  # Chevrolet Avalanche Crew Cab 2012
    66: ("H", "E"),  # Chevrolet Cobalt SS 2010
    67: ("I", "D"),  # Chevrolet Malibu Hybrid Sedan 2010
    68: ("R", "F"),  # Chevrolet TrailBlazer SS 2009
    69: ("O", "P"),  # Chevrolet Silverado 2500HD Regular Cab 2012
    70: ("O", "P"),  # Chevrolet Silverado 1500 Classic Extended Cab 2007
    71: ("O", "V"),  # Chevrolet Express Van 2007
    72: ("S", "E"),  # Chevrolet Monte Carlo Coupe 2007
    73: ("I", "D"),  # Chevrolet Malibu Sedan 2007
    74: ("O", "P"),  # Chevrolet Silverado 1500 Extended Cab 2012
    75: ("O", "P"),  # Chevrolet Silverado 1500 Regular Cab 2012
    76: ("S", "F"),  # Chrysler Aspen SUV 2009
    77: ("I", "T"),  # Chrysler Sebring Convertible 2010
    78: ("S", "M"),  # Chrysler Town and Country Minivan 2012
    79: ("R", "D"),  # Chrysler 300 SRT-8 2010
    80: ("J", "T"),  # Chrysler Crossfire Convertible 2008
    81: ("C", "T"),  # Chrysler PT Cruiser Convertible 2008
    82: ("E", "W"),  # Daewoo Nubira Wagon 2002
    83: ("C", "W"),  # Dodge Caliber Wagon 2012
    84: ("C", "W"),  # Dodge Caliber Wagon 2007
    85: ("S", "M"),  # Dodge Caravan Minivan 1997
    86: ("O", "Q"),  # Dodge Ram Pickup 3500 Crew Cab 2010
    87: ("O", "Q"),  # Dodge Ram Pickup 3500 Quad Cab 2009
    88: ("O", "K"),  # Dodge Sprinter Cargo Van 2009
    89: ("I", "F"),  # Dodge Journey SUV 2012
    90: ("O", "Q"),  # Dodge Dakota Crew Cab 2010
    91: ("O", "P"),  # Dodge Dakota Club Cab 2007
    92: ("S", "W"),  # Dodge Magnum Wagon 2008
    93: ("R", "E"),  # Dodge Challenger SRT8 2011
    94: ("S", "F"),  # Dodge Durango SUV 2012
    95: ("S", "F"),  # Dodge Durango SUV 2007
    96: ("S", "D"),  # Dodge Charger Sedan 2012
    97: ("R", "D"),  # Dodge Charger SRT-8 2009
    98: ("C", "B"),  # Eagle Talon Hatchback 1998
    99: ("N", "B"),  # FIAT 500 Abarth 2012
    100: ("M", "T"),  # FIAT 500 Convertible 2012
    101: ("X", "E"),  # Ferrari FF Coupe 2012
    102: ("X", "T"),  # Ferrari California Convertible 2012
    103: ("X", "T"),  # Ferrari 458 Italia Convertible 2012
    104: ("X", "E"),  # Ferrari 458 Italia Coupe 2012
    105: ("X", "D"),  # Fisker Karma Sedan 2012
    106: ("O", "Q"),  # Ford F-450 Super Duty Crew Cab 2012
    107: ("R", "T"),  # Ford Mustang Convertible 2007
    108: ("S", "M"),  # Ford Freestar Minivan 2007
    109: ("F", "F"),  # Ford Expedition EL SUV 2009
    110: ("I", "G"),  # Ford Edge SUV 2012
    111: ("O", "P"),  # Ford Ranger SuperCab 2011
    112: ("X", "E"),  # Ford GT Coupe 2006
    113: ("O", "P"),  # Ford F-150 Regular Cab 2012
    114: ("O", "P"),  # Ford F-150 Regular Cab 2007
    115: ("C", "D"),  # Ford Focus Sedan 2007
    116: ("O", "V"),  # Ford E-Series Wagon Van 2012
    117: ("E", "D"),  # Ford Fiesta Sedan 2012
    118: ("I", "F"),  # GMC Terrain SUV 2012
    119: ("O", "V"),  # GMC Savana Van 2012
    120: ("F", "F"),  # GMC Yukon Hybrid SUV 2012
    121: ("S", "F"),  # GMC Acadia SUV 2012
    122: ("O", "P"),  # GMC Canyon Extended Cab 2012
    123: ("M", "T"),  # Geo Metro Convertible 1993
    124: ("O", "Q"),  # HUMMER H3T Crew Cab 2010
    125: ("O", "Q"),  # HUMMER H2 SUT Crew Cab 2009
    126: ("S", "M"),  # Honda Odyssey Minivan 2012
    127: ("S", "M"),  # Honda Odyssey Minivan 2007
    128: ("I", "E"),  # Honda Accord Coupe 2012
    129: ("I", "D"),  # Honda Accord Sedan 2012
    130: ("C", "B"),  # Hyundai Veloster Hatchback 2012
    131: ("I", "F"),  # Hyundai Santa Fe SUV 2012
    132: ("C", "F"),  # Hyundai Tucson SUV 2012
    133: ("S", "F"),  # Hyundai Veracruz SUV 2012
    134: ("I", "D"),  # Hyundai Sonata Hybrid Sedan 2012
    135: ("C", "D"),  # Hyundai Elantra Sedan 2007
    136: ("E", "D"),  # Hyundai Accent Sedan 2012
    137: ("P", "D"),  # Hyundai Genesis Sedan 2012
    138: ("I", "D"),  # Hyundai Sonata Sedan 2012
    139: ("C", "D"),  # Hyundai Elantra Touring Hatchback 2012
    140: ("S", "D"),  # Hyundai Azera Sedan 2012
    141: ("U", "E"),  # Infiniti G Coupe IPL 2012
    142: ("P", "F"),  # Infiniti QX56 SUV 2011
    143: ("S", "F"),  # Isuzu Ascender SUV 2008
    144: ("L", "E"),  # Jaguar XK XKR 2012
    145: ("C", "F"),  # Jeep Patriot SUV 2012
    146: ("J", "J"),  # Jeep Wrangler SUV 2012
    147: ("I", "F"),  # Jeep Liberty SUV 2012
    148: ("S", "F"),  # Jeep Grand Cherokee SUV 2012
    149: ("C", "F"),  # Jeep Compass SUV 2012
    150: ("X", "E"),  # Lamborghini Reventon Coupe 2008
    151: ("X", "E"),  # Lamborghini Aventador Coupe 2012
    152: ("X", "E"),  # Lamborghini Gallardo LP 570-4 Superleggera 2012
    153: ("X", "E"),  # Lamborghini Diablo Coupe 2001
    154: ("L", "F"),  # Land Rover Range Rover SUV 2012
    155: ("P", "F"),  # Land Rover LR2 SUV 2012
    156: ("L", "D"),  # Lincoln Town Car Sedan 2011
    157: ("N", "N"),  # MINI Cooper Roadster Convertible 2012
    158: ("W", "T"),  # Maybach Landaulet Convertible 2012
    159: ("C", "F"),  # Mazda Tribute SUV 2011
    160: ("X", "E"),  # McLaren MP4-12C Coupe 2012
    161: ("P", "T"),  # Mercedes-Benz 300-Class Convertible 1993
    162: ("P", "D"),  # Mercedes-Benz C-Class Sedan 2012
    163: ("L", "E"),  # Mercedes-Benz SL-Class Coupe 2009
    164: ("P", "D"),  # Mercedes-Benz E-Class Sedan 2012
    165: ("L", "D"),  # Mercedes-Benz S-Class Sedan 2012
    166: ("O", "K"),  # Mercedes-Benz Sprinter Van 2012
    167: ("C", "D"),  # Mitsubishi Lancer Sedan 2012
    168: ("C", "D"),  # Nissan Leaf Hatchback 2012
    169: ("O", "V"),  # Nissan NV Passenger Van 2012
    170: ("C", "G"),  # Nissan Juke Hatchback 2012
    171: ("C", "E"),  # Nissan 240SX Coupe 1998
    172: ("E", "E"),  # Plymouth Neon Coupe 1999
    173: ("L", "D"),  # Porsche Panamera Sedan 2012
    174: ("O", "K"),  # Ram C/V Cargo Van Minivan 2012
    175: ("W", "T"),  # Rolls-Royce Phantom Drophead Coupe Convertible 2012
    176: ("W", "D"),  # Rolls-Royce Ghost Sedan 2012
    177: ("W", "D"),  # Rolls-Royce Phantom Sedan 2012
    178: ("E", "D"),  # Scion xD Hatchback 2012
    179: ("X", "T"),  # Spyker C8 Convertible 2009
    180: ("X", "E"),  # Spyker C8 Coupe 2009
    181: ("E", "D"),  # Suzuki Aerio Sedan 2007
    182: ("C", "D"),  # Suzuki Kizashi Sedan 2012
    183: ("E", "D"),  # Suzuki SX4 Hatchback 2012
    184: ("E", "D"),  # Suzuki SX4 Sedan 2012
    185: ("P", "D"),  # Tesla Model S Sedan 2012
    186: ("F", "F"),  # Toyota Sequoia SUV 2012
    187: ("I", "D"),  # Toyota Camry Sedan 2012
    188: ("C", "D"),  # Toyota Corolla Sedan 2012
    189: ("S", "F"),  # Toyota 4Runner SUV 2012
    190: ("C", "D"),  # Volkswagen Golf Hatchback 2012
    191: ("C", "D"),  # Volkswagen Golf Hatchback 1991
    192: ("C", "D"),  # Volkswagen Beetle Hatchback 2012
    193: ("P", "D"),  # Volvo C30 Hatchback 2012
    194: ("I", "D"),  # Volvo 240 Sedan 1993
    195: ("P", "F"),  # Volvo XC90 SUV 2007
    196: ("M", "T"),  # smart fortwo Convertible 2012
}


def get_acriss(class_id_1based: int) -> tuple[str, str]:
    if class_id_1based not in STANFORD_TO_ACRISS:
        raise KeyError(f"No ACRISS mapping for Stanford class {class_id_1based}")
    return STANFORD_TO_ACRISS[class_id_1based]


def category_label(code: str) -> str:
    return CATEGORIES[code]


def type_label(code: str) -> str:
    return TYPES[code]
