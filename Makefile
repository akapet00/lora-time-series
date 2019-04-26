all: src/data/LORA_data.csv

clean:
	rm -f src/data/raw/*

get_raw_data:
	python src/data.py $(SVEBOLLE_URL) src/data/raw/