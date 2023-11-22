# Access Methods

## Overview
This documentation provides guidance on accessing the web service hosted at [http://3.142.238.44:5000](http://3.142.238.44:5000). The service responds differently based on the HTTP method used.

### GET Method
- **Usage**: Simply sending a GET request will yield a "Hello World!" response.
  
### POST Method
- **Usage**: To utilize the POST method, include JSON data in the body of the request. For instance:

```json
{
    "year": 2021,
    "month": 1
}
```
This request will return a forecast result, such as `23.2086`.

# Conceptual Framework

## 1. Dataset Processing
The dataset, focusing on real-world traffic accidents, undergoes rigorous quality checks in `1_Data_Check.ipynb`. This ensures the elimination of duplications or double-counted entries.

Certain elements in the original dataset, like the ones listed below, are not directly usable as they pertain to our target variable, 'Value':
- `VERAEND_VORMONAT_PROZENT` (Change_From_Previous_Month_Percentage)
- `VERAEND_VORJAHRESMONAT_PROZENT` (Change_From_Previous_Year_Month_Percentage)

In response, we've removed these and expanded our feature set with alternatives like:
- `VORJAHRESWERT` (Previous_Year_Value)
- `ZWOELF_MONATE_MITTELWERT` (Twelve_Month_Average)

Remaining features include:
- `MONATSZAHL` (Category)
- `AUSPRAEGUNG` (Accident-type)
- `JAHR` (Year)
- `MONAT` (Month)
- `WERT` (Value)

In these columns, 'Value' is the target. Feature expansion was conducted with considerable flexibility, utilizing various methodological adjustments like the size of the time window, choice of aggregation methods (mean, standard deviation, etc.), and selecting values from the same month in previous years.

## 2. Model Selection

The rationale behind not employing complex models stems from the data's structure. Grouping 'Category' and 'Accident-type' together results in approximately seven groups, each sharing the same timeline but with only about 200 data rows, rendering them unsuitable for complex models like neural networks.

Moreover, a correlation analysis revealed high inter-category correlations within traffic accident types. Hence, in 2_Feature_Expansion_and_Modeling, both individual group data and aggregated group data were utilized to predict the 'Alkoholunfälle - insgesamt' group value. Interestingly, using individual group data yielded slightly better results, leading to the selection and preservation of this approach for the final model.
## 3. Future Directions

### Feature Expansion and Model Tuning
Given the time constraints, basic parameter settings were used for feature expansion, and a selection of models were tested to choose the best performer. However, integrating external datasets could significantly enhance the predictive accuracy by incorporating variables like temperature, humidity, weather conditions, holidays, policy implementations, visibility, rainfall, and daylight duration.

### Constraint Implementation
In scenarios with sufficient data, constraints such as ensuring the 'insgesamt' (total) accident type is always greater than or equal to other types can be incorporated into the loss function. This approach, drawn from my master's thesis, aims to correct erroneous predictions and improve model performance.

# Package Versions

Below are the versions of the primary packages used, along with their dependencies and the corresponding Python version.

- Anaconda: Python 3.9.16

- Scikit-Learn: `pip install scikit-learn==1.3.2`
  - Dependencies: numpy-1.26.2, scipy-1.11.3, threadpoolctl-3.2.0

- Pandas: `pip install pandas==2.1.3`
  - Dependencies: python-dateutil-2.8.2, pytz-2023.3.post1, six-1.16.0, tzdata-2023.3

- Matplotlib: `pip install matplotlib==3.8.2`
  - Dependencies: contourpy-1.2.0, cycler-0.12.1, fonttools-4.44.3, importlib-resources-6.1.1, kiwisolver-1.4.5, pillow-10.1.0, pyparsing-3.1.1

- Jinja2: `pip install Jinja2==3.1.2`
    - Dependencies: MarkupSafe-2.1.3

- Flask: `pip install Flask==3.0.0`
    - Dependencies: Werkzeug-3.0.1 blinker-1.7.0 click-8.1.7 itsdangerous-2.1.2