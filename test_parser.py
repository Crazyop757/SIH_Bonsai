from fra_parser import parse_file

# # Parse a CSV file
# fra1 = parse_file("data/omicron_sample.csv", vendor="Omicron")
# print("CSV Parsed:", fra1)

# Parse an XML file
fra2 = parse_file("megger_sample.xml", vendor="Megger")
print("XML Parsed:", fra2)

# # Parse a TXT file (if available)
# fra3 = parse_file("data/doble_sample.txt", vendor="Doble")
# print("TXT Parsed:", fra3)
