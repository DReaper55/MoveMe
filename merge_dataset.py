import json

# Load orders data
with open('orders_data.json', 'r') as orders_file:
    orders_data = json.load(orders_file)

# Load drivers data
with open('drivers_data.json', 'r') as drivers_file:
    drivers_data = json.load(drivers_file)

# Create a dictionary to store the merged dataset
merged_dataset = {"dataset": []}

# Iterate through each order
for order in orders_data["orders"]:
    order_id = order["OrderID"]

    print(order_id)

    # Find corresponding drivers for the order
    matching_drivers = [driver for driver in drivers_data["drivers"]]

    # Create a dictionary for the order and associated drivers
    order_with_drivers = {
        "order": order,
        "drivers": matching_drivers
    }

    # Add the order_with_drivers to the merged dataset
    merged_dataset["dataset"].append(order_with_drivers)

    # Update CurrentOrders for each selected driver
    # for driver in matching_drivers:
    #     driver["CurrentOrders"] += 1

# Save the merged dataset to a new JSON file
with open('merged_dataset.json', 'w') as merged_file:
    json.dump(merged_dataset, merged_file, indent=2)

print("Merged dataset saved to merged_dataset.json")
