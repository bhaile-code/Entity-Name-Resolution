"""
Generate difficult variations to augment sample_data_500.csv
"""
import csv

# Read existing data
existing_names = []
with open('sample_data_500.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        existing_names.append(row['company_name'])

print(f'Loaded {len(existing_names)} existing company names')

# Create difficult variations
difficult_variations = []

# Category 1: Aggressive Typos (60-80% confidence expected)
difficult_variations.extend([
    'Microsft Corporation',  # Microsoft
    'Gogle Inc',  # Google
    'Amazn.com',  # Amazon
    'Appel Computer',  # Apple
    'Oralce Systems',  # Oracle
    'Nettflix Media',  # Netflix
    'Paypal Holdings',  # PayPal (case)
    'LinkedIN Corp',  # LinkedIn (case)
    'Nvidea Corporation',  # Nvidia
    'Salesfarce Inc',  # Salesforce
    'Twiter Inc',  # Twitter
    'Cisko Systems',  # Cisco
    'Samung Electronics',  # Samsung
    'Toyata Motors',  # Toyota
    'Honда Motor',  # Honda (Cyrillic 'a')
    'Volksagen Group',  # Volkswagen
    'Mircrosoft Corp',  # Microsoft (another typo)
    'Gooogle LLC',  # Google (double o)
    'Teslla Motors',  # Tesla
    'Spotfy Music',  # Spotify
])

# Category 2: Heavy Abbreviations & Informal Names (65-85% confidence)
difficult_variations.extend([
    'MSFT',  # Microsoft ticker
    'AAPL',  # Apple ticker
    'GOOGL',  # Google ticker
    'AMZN',  # Amazon ticker
    'TSLA',  # Tesla ticker
    'FB',  # Facebook old ticker
    'NFLX',  # Netflix ticker
    'ORCL',  # Oracle ticker
    'Big Blue',  # IBM nickname
    'The Zon',  # Amazon slang
    'MS Corp USA',  # Microsoft
    'Intl Business Machines',  # IBM
    'Zuck Company',  # Meta/Facebook
    'GE Co',  # General Electric
    'BAC',  # Bank of America ticker
    'JPM',  # JPMorgan ticker
    'WFC',  # Wells Fargo ticker
    'WMT',  # Walmart ticker
    'TGT',  # Target ticker
    'HD',  # Home Depot ticker
])

# Category 3: Challenging Phonetic Matches (70-90% confidence)
difficult_variations.extend([
    'Microsawft Corp',  # Microsoft
    'Gugle Inc',  # Google
    'Amajon.com',  # Amazon
    'Fazebook Inc',  # Facebook
    'Linkd In',  # LinkedIn
    'NetFlix Inc',  # Netflix (camelCase)
    'Pay Pal Holdings',  # PayPal (space)
    'SalesForce Inc',  # Salesforce (camelCase)
    'Nvidiya Corp',  # Nvidia
    'Orakel Systems',  # Oracle
    'Spotifye Music',  # Spotify
    'Tezla Motors',  # Tesla
    'Sisco Systems',  # Cisco
    'Toyoda Motors',  # Toyota (founder's name)
    'Macdonalds Corp',  # McDonalds
])

# Category 4: Ambiguous Multi-Company Names (should NOT group, <70%)
difficult_variations.extend([
    'American Standard',  # NOT American Express/Airlines
    'American General Insurance',  # NOT American Express/Airlines
    'Delta Dental',  # NOT Delta Airlines
    'Delta Faucet Company',  # NOT Delta Airlines
    'Target Marketing Group',  # NOT Target Corporation
    'Target Media',  # NOT Target Corporation
    'Oracle Financial Services',  # Ambiguous - subsidiary or different?
    'Apple Records Limited',  # NOT Apple Inc (Beatles label)
    'Amazon Logistics LLC',  # Subsidiary or different?
    'Meta Financial Group',  # NOT Meta Platforms!
    'United Healthcare',  # NOT United Airlines
    'United Rentals Inc',  # NOT United Airlines
    'General Mills Inc',  # NOT General Motors/GE
    'General Dynamics Corp',  # NOT General Motors/GE
    'Continental Airlines',  # Merged, but test
    'Continental Tire',  # NOT Continental Airlines
])

# Category 5: Similar-Sounding Different Companies (should NOT group, <60%)
difficult_variations.extend([
    'Zoom Telephonics Inc',  # NOT Zoom Video!
    'Domino Printing Sciences',  # NOT Dominos Pizza
    'Adobe Rent-A-Car',  # NOT Adobe Systems
    'SpaceX',  # NOT X Corp/Twitter (both Musk)
    'Space Exploration Technologies',  # SpaceX full name
    'Chase Bank NA',  # Part of JPMorgan but test
    'Morgan Corporation',  # NOT Morgan Stanley/JPMorgan
    'First American Financial',  # NOT American Express/Airlines
    'Delta Community Credit Union',  # NOT Delta Airlines
    'American Family Insurance',  # NOT American Express/Airlines
])

# Category 6: Merger/Acquisition Relationships (60-85% confidence)
difficult_variations.extend([
    'Hewlett Packard Enterprise',  # Split from HP
    'HPE',  # Hewlett Packard Enterprise ticker
    'eBay Inc',  # Used to own PayPal
    'YouTube LLC',  # Google subsidiary
    'Instagram Inc',  # Meta subsidiary
    'WhatsApp Inc',  # Meta subsidiary
    'Whole Foods Market',  # Amazon subsidiary
    'GitHub Inc',  # Microsoft subsidiary
    'GitHub',  # Microsoft subsidiary
    'Oculus VR',  # Meta subsidiary
    'Beats Electronics',  # Apple subsidiary
    'Waze Mobile',  # Google subsidiary
    'Nest Labs',  # Google subsidiary
])

# Category 7: International Variations (70-90% confidence)
difficult_variations.extend([
    'Volkswagen Aktiengesellschaft',  # VW full name
    'Bayerische Motoren Werke AG',  # BMW full name
    'Société Générale',  # French bank
    'Societe Generale SA',  # Without accent
    'BP p.l.c.',  # British Petroleum
    'HSBC Holdings plc',  # Bank
    'Hongkong and Shanghai Banking Corporation',  # HSBC full name
    'SAP SE',  # German software
    'SAP AG',  # Old legal form
    'Daimler-Benz AG',  # Historical Mercedes
    'Fiat Chrysler Automobiles',  # Before Stellantis merger
    'Stellantis N.V.',  # After FCA merger
])

# Category 8: Extra Noise & Geographic Markers (75-90% confidence)
difficult_variations.extend([
    'Microsoft Corporation, Redmond WA',
    'Apple Inc. Cupertino California',
    'Amazon.com Inc (Seattle)',
    'Google LLC - Mountain View',
    'Tesla Inc., Austin TX',
    'Meta Platforms Inc, Menlo Park',
    'Oracle Corporation, Austin Texas',
    'IBM Corp, Armonk NY',
    'Intel Corporation Santa Clara',
    'Cisco Systems San Jose CA',
    'The Coca Cola Company Atlanta',
    'Nike Inc Beaverton Oregon',
    'Walmart Inc Bentonville Arkansas',
    'Boeing Company Seattle WA',
    'Microsoft Corp USA',
])

# Category 9: Historical/Old Names (60-80% confidence)
difficult_variations.extend([
    'BackRub Inc',  # Original Google name - VERY HARD
    'Research In Motion',  # BlackBerry
    'RIM',  # Research In Motion
    'BlackBerry Ltd',  # RIM rebrand
    'Philip Morris Companies',  # Altria
    'Altria Group Inc',  # Philip Morris rebrand
    'Andersen Consulting',  # Accenture
    'Accenture plc',  # Andersen rebrand
    'Quantum Computer Services',  # AOL
    'AOL Inc',  # America Online
    'America Online',  # AOL
    'Time Warner Inc',  # Before AT&T merger
    'Apple Computer Company',  # Historical Apple
])

# Category 10: New Companies - Different but share words
difficult_variations.extend([
    # Healthcare/Pharma (new)
    'CVS Health Corporation',
    'CVS Pharmacy',
    'CVS Caremark',
    'Walgreens Boots Alliance',
    'Walgreens Co',
    'Walgreens Pharmacy',
    'UnitedHealth Group',  # NOT United Airlines
    'UnitedHealthcare',  # NOT United Airlines
    'Anthem Inc',
    'Humana Inc',

    # Retail (new challenging ones)
    'Best Buy Co Inc',
    'Best Buy',
    'Best Buy Electronics',
    'Circuit City',  # Out of business, but test
    'Radio Shack Corporation',
    'RadioShack',
    'Gap Inc',
    'The Gap',
    'Old Navy',  # Gap subsidiary

    # Food/Beverage (more variations)
    'Yum! Brands Inc',  # KFC, Taco Bell parent
    'Yum Brands',
    'KFC Corporation',
    'Kentucky Fried Chicken',
    'Taco Bell Corp',
    'Dunkin Brands',
    'Dunkin Donuts',
    'Dominos Pizza Inc',
    'Dominoes Pizza',

    # Tech (more ambiguous)
    'Oracle Corp USA',  # Oracle
    'Oracle America Inc',  # Oracle subsidiary
    'Sun Microsystems',  # Acquired by Oracle
    'Red Hat Inc',  # IBM subsidiary
    'VMware Inc',  # Broadcom subsidiary
    'Salesforce.com',  # Salesforce (already exists but testing)

    # Finance (more)
    'Charles Schwab Corporation',
    'Charles Schwab',
    'TD Ameritrade',
    'E*TRADE Financial',
    'Fidelity Investments',
    'Vanguard Group',

    # Airlines/Transport
    'JetBlue Airways',
    'JetBlue',
    'Spirit Airlines',
    'Frontier Airlines',
    'Alaska Airlines',
])

print(f'\nCreated {len(difficult_variations)} difficult variations')
print(f'Total names will be: {len(existing_names) + len(difficult_variations)}')

# Combine and write to new CSV
all_names = existing_names + difficult_variations

with open('sample_data_500.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['company_name'])
    for name in all_names:
        writer.writerow([name])

print(f'\nWrote {len(all_names)} names to sample_data_500.csv')
print('\nSample of new variations added:')
for i, name in enumerate(difficult_variations[:25]):
    print(f'  {i+1}. {name}')
