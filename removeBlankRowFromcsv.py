import csv

filename = "sample/all.csv"
cleanedFile = "sample/allcleaned.csv"
with open(filename) as in_file:
    with open(cleanedFile, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(in_file):
            if any(field.strip() for field in row):
                writer.writerow(row)