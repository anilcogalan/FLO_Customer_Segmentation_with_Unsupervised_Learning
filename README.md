# FLO Customer Segmentation with Unsupervised Learning

<p align="center">
  <img width="300" height="600" src="https://user-images.githubusercontent.com/61653147/235902153-e73bcb93-9df6-40e7-a016-873b76488c8b.jpg">
</p>

## Business Problem

FLO segments its customers and according to these segments
Wants to define marketing strategies. for this
As a result, the behavior of customers will be defined and this
Groups will be formed according to clusters in behaviors

## Data Set

The dataset consists of information obtained from the past shopping behaviors of customers who made their last purchases on OmniChannel (both online and offline) in 2020 - 2021.

| Column Name                       | Description                                                                                                      |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
| master_id                        | Unique customer number                                                                                            |
| order_channel                    | The channel used for the shopping platform (Android, iOS, Desktop, Mobile, Offline)                              |
| last_order_channel               | The channel used for the last shopping experience                                                                  |
| first_order_date                 | The date of the customer's first shopping experience                                                              |
| last_order_date                  | The date of the customer's last shopping experience                                                               |
| last_order_date_online           | The date of the customer's last online shopping experience                                                       |
| last_order_date_offline          | The date of the customer's last offline shopping experience                                                      |
| order_num_total_ever_online      | The total number of online shopping experiences by the customer                                                  |
| order_num_total_ever_offline     | The total number of offline shopping experiences by the customer                                                 |
| customer_value_total_ever_offline| The total amount paid by the customer for offline shopping experiences                                           |
| customer_value_total_ever_online | The total amount paid by the customer for online shopping experiences                                            |
| interested_in_categories_12      | The list of categories that the customer has shopped in the last 12 months                                       |
| store_type                       | Indicates three different companies. If a customer made a purchase from company A and also from B, it is written as A, B. |
