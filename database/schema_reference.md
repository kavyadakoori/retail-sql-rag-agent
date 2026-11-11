# Database Schema Reference

This document describes the MySQL schema used by the retail agent assignment project.

## Database

- Database name: `retail_agent_assignment`

## Tables

### 1. stores
Stores basic information about retail outlets.

| Column | Type | Constraints | Description |
| --- | --- | --- | --- |
| store_id | VARCHAR(20) | PRIMARY KEY | Unique identifier for each store |
| store_name | VARCHAR(100) | | Name of the store |
| region | VARCHAR(50) | | Geographic region |
| city | VARCHAR(50) | | City where the store is located |
| store_type | VARCHAR(50) | | Type of store, such as Supermarket or Convenience |

### 2. products
Stores product catalog details.

| Column | Type | Constraints | Description |
| --- | --- | --- | --- |
| product_id | VARCHAR(20) | PRIMARY KEY | Unique identifier for each product |
| product_name | VARCHAR(100) | | Name of the product |
| category | VARCHAR(50) | | Main product category |
| sub_category | VARCHAR(50) | | Product sub-category |
| base_price | DECIMAL(10, 2) | | Base selling price |

### 3. returns
Tracks product returns.

| Column | Type | Constraints | Description |
| --- | --- | --- | --- |
| return_id | VARCHAR(20) | PRIMARY KEY | Unique identifier for each return |
| order_id | VARCHAR(20) | | Associated order identifier |
| return_date | DATE | | Date when the return was made |
| return_reason | VARCHAR(200) | | Reason given for the return |

### 4. customers
Stores customer profile information.

| Column | Type | Constraints | Description |
| --- | --- | --- | --- |
| customer_id | VARCHAR(20) | PRIMARY KEY | Unique identifier for each customer |
| customer_segment | VARCHAR(50) | | Customer segment such as Regular or Premium |
| signup_date | DATE | | Date the customer signed up |
| preferred_channel | VARCHAR(100) | | Preferred sales or communication channel |
| city | VARCHAR(50) | | Customer city |

### 5. sales_transactions
Stores sales order transaction details.

| Column | Type | Constraints | Description |
| --- | --- | --- | --- |
| order_id | VARCHAR(20) | PRIMARY KEY | Unique identifier for each sales order |
| order_date | DATE | | Date the order was placed |
| store_id | VARCHAR(20) | | Reference to the store |
| product_id | VARCHAR(20) | | Reference to the product |
| customer_id | VARCHAR(20) | | Reference to the customer |
| sales_channel | VARCHAR(100) | | Channel used for the sale |
| units_sold | INT | | Number of units sold |
| unit_price | DECIMAL(10, 2) | | Price per unit |
| discount_pct | DECIMAL(5, 2) | | Discount percentage applied |
| payment_status | VARCHAR(50) | | Payment status of the order |
| delivery_status | VARCHAR(50) | | Current delivery status |

## Notes

- The schema is intentionally simple and uses string-based identifiers for the main entities.
- The `sales_transactions` table is the primary fact table for retail analytics questions.
- The `returns` table records returns by `order_id`; it does not contain a returned-quantity column.

## Relationships

Use the following documented join paths when a question needs data from
related tables:

- `sales_transactions.store_id = stores.store_id`
- `sales_transactions.product_id = products.product_id`
- `sales_transactions.customer_id = customers.customer_id`
- `sales_transactions.order_id = returns.order_id`

These relationships can be joined directly or as part of a multi-table join.
For example, sales or returns can be grouped by store by joining
`sales_transactions` to `stores`; a store identifier does not need to be
present in `returns`.
