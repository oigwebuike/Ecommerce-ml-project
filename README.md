## An Ecommerce Site's Annual Spending Prediction ML project deployed using Heroku and Docker
- This end-to-end machine learning (ML) project is designed to train an ML model using two different algorithms, to predict the amount a customer spends yearly on the site, given different independent variables like time spent on the website and length of membership of the customer.
- The linear regression model aims to predict the amount a customer spends annually on an ecommerce site. The model will be trained using two different algorithms (sklearn and xgboost), and will use independent variables such as time spent on the website and length of customer membership to make predictions.
-  The project uses a sample dataset and a Flask app to serve as the backend for the deployed model.

### Software and tools required
- 1. A Github account: [GitHub](https://github.com/) and [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)
- 2. An IDE: [VSCode](https://code.visualstudio.com/) or [PyCharm](https://www.jetbrains.com/pycharm/)
- 3. A Heroku account: [Heroku](https://id.heroku.com/login)
- 4. A Docker account: [Docker](https://www.docker.com/)


### Requirements
- Python 3.x
- Flask
- pandas
- numpy
- matplotlib
- xgboost
- seaborn
- gunicorn
- scikit-learn
- jupyterlab
### Create a new environment for the project

```
conda create -p my_env python==3.8 -y
```

To activate the new environment created run:
```
conda activate my_env
```

### File Structure
- `app.py` : Flask app file that runs the project
- `project_nb.ipynb` : Jupyter notebook containing the code for the project
- `Dockerfile` : Docker file used to spin up containers for the project
- `data` : folder containing the data used in the project
- `template` : folder containing the html file used as frontend page for the project
- `models` : folder containing the models used in the project
- `requirements.txt` : list of dependencies for the project



### Running the Project
1. Clone the repository
2. Install the dependencies by running
```
 pip install -r requirements.txt
```
3. Open the `project_nb.ipynb` in Jupyter notebook
4. Run the cells in the notebook in order

#### Project deployed at [ecommerce-spending-estimate](https://ecommerce-spending-estimate.herokuapp.com/)




## Conclusion
The model will be trained and tested using the provided data, and the performance of the two algorithms will be compared. The results of the prediction will be visualized, and the best algorithm will be chosen based on the accuracy of the predictions. This project demonstrates how to train an ML model and deploy it using Heroku and Docker. The model is trained using a sample dataset, and the Flask app serves as the backend for the deployed model. The Docker image includes all the necessary dependencies and configurations, making it easy to deploy the model to Heroku.
