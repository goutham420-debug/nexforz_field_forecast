from datetime import date
sample_reading = {
	"date":date.today().isoformat(),
	"temperature_c": 22.4,
	"humidity_pct":88.5,
	"co2_ppm":950,
	"yield_kg":12.3,
}
print("Polyhouse sensor snapshot:")
for key, value in sample_reading.items():
	print(f" {key}: {value}")
assert sample_reading["humidity_pct"]>80,"Oyster mushrooms need high humidity"
print("Environment OK.")