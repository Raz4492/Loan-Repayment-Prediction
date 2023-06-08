# -*- coding: utf-8 -*-
import io
from base64 import b64encode

from IPython.core.display import HTML
from bokeh.io import output_file, show
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, render_template, request, url_for, session
import pickle
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, gridplot
from bokeh.transform import factor_cmap
from bokeh.embed import components
import sqlite3
import lime
import lime.lime_tabular
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def predict():
    # Getting the data from the form
    current_loan_amount = request.form['Current Loan Amount']
    term = request.form['Term']
    credit_score = request.form['Credit Score']
    annual_income = request.form['Annual Income']
    home_ownership = request.form['Home Ownership']
    monthly_debt = request.form['Monthly Debt']
    years_credit_hist = request.form['Years of Credit History']
    months_since_last_delinquent = request.form['Months Since Last Delinquent']
    number_of_open_accounts = request.form['Number of Open Accounts']
    number_credit_prob = request.form['Number of Credit Problems']
    current_credit_balance = request.form['Current Credit Balance']
    max_open_credit = request.form['Maximum Open Credits']
    bankruptcies = request.form['Bankruptcies']
    tax_liens = request.form['Tax Liens']
    years_current_job = request.form['Years in current job']

    #  creating a json object to hold the data from the form
    input_data = [{

        'current_loan_amount': current_loan_amount,
        'term': term,
        'credit_score': credit_score,
        'annual_income': annual_income,
        'home_ownership': home_ownership,
        'monthly_debt': monthly_debt,
        'years_credit_hist': years_credit_hist,
        'months_since_last_delinquent': months_since_last_delinquent,
        'number_of_open_accounts': number_of_open_accounts,
        'number_credit_prob': number_credit_prob,
        'current_credit_balance': current_credit_balance,
        'max_open_credit': max_open_credit,
        'bankruptcies': bankruptcies,
        'tax_liens': tax_liens,
        'years_current_job': years_current_job
    }]

    dataset = pd.DataFrame(input_data)
    # print(dataset)
    dataset = dataset.rename(columns={
        'current_loan_amount': 'Current Loan Amount',
        'term': 'Term',
        'credit_score': 'Credit Score',
        'annual_income': 'Annual Income',
        'home_ownership': 'Home Ownership',
        'monthly_debt': 'Monthly Debt',
        'years_credit_hist': 'Years of Credit History',
        'months_since_last_delinquent': 'Months since last delinquent',
        'number_of_open_accounts': 'Number of Open Accounts',
        'number_credit_prob': 'Number of Credit Problems',
        'current_credit_balance': 'Current Credit Balance',
        'max_open_credit': 'Maximum Open Credit',
        'bankruptcies': 'Bankruptcies',
        'tax_liens': 'Tax Liens',
        'years_current_job': 'Years'
    })

    dataset[['Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
             'Home Ownership', 'Monthly Debt', 'Years of Credit History',
             'Months since last delinquent', 'Number of Open Accounts',
             'Number of Credit Problems', 'Current Credit Balance',
             'Maximum Open Credit', 'Bankruptcies', 'Tax Liens', 'Years']] = dataset[['Current Loan Amount',
                                                                                      'Term', 'Credit Score',
                                                                                      'Annual Income', 'Home Ownership',
                                                                                      'Monthly Debt',
                                                                                      'Years of Credit History',
                                                                                      'Months since last delinquent',
                                                                                      'Number of Open Accounts',
                                                                                      'Number of Credit Problems',
                                                                                      'Current Credit Balance',
                                                                                      'Maximum Open Credit',
                                                                                      'Bankruptcies', 'Tax Liens',
                                                                                      'Years']].astype(float)

    datasets = dataset[[
        'Current Loan Amount', 'Term', 'Credit Score', 'Annual Income',
        'Home Ownership', 'Monthly Debt', 'Years of Credit History',
        'Months since last delinquent', 'Number of Open Accounts',
        'Number of Credit Problems', 'Current Credit Balance',
        'Maximum Open Credit', 'Bankruptcies', 'Tax Liens', 'Years'
    ]]
    with open('CatboostModel.pkl', 'rb') as f:
        model = pickle.load(f)

    classifier = model.predict_proba(datasets)
    predictions = [item for sublist in classifier for item in sublist]
    predict0 = round(predictions[0], 2) * 100
    predict1 = round(predictions[1], 2) * 100

    def my_func(p,q):
        r = p,'%',' _________________________________________________________________', q,'%'
        return r
    predict = my_func(predict0,predict1)
    colors = ['#1F77B4', '#FF7F0E']
    loan_status = ['Fully Paid', 'Charged Off']
    source = ColumnDataSource(
        data=dict(loan_status=loan_status, predictions=predictions))

    p = figure(x_range=loan_status, plot_height=450,
               toolbar_location=None,  plot_width=1000, background_fill_color='#F0F0F0', border_fill_color='#FAFCFF')
    p.vbar(x='loan_status', top='predictions', width=0.2, source=source, legend="loan_status",
           line_color='black', fill_color=factor_cmap('loan_status', palette=colors, factors=loan_status))

    p.xgrid.grid_line_color = None
    p.y_range.start = 0.1
    p.y_range.end = 0.9
    p.legend.orientation = "horizontal"
    p.legend.location = "top_center"
    p.xaxis.axis_label = 'Investment Status'
    p.yaxis.axis_label = 'Investment Repayment Probability'
    script, div = components(p)

    feature_names = dataset.columns
    with open('X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)

    test = dataset
    predict_fn = lambda x: model.predict_proba(x).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=feature_names,
                                                           class_names=['Fully Paid','Charged Off'], kernel_width=1)

    exp = explainer.explain_instance(test.iloc[0], predict_fn, num_features=len(feature_names))
    html_data = exp.as_html(predict_proba=True, show_predicted_value=False)
    html_data=HTML(data=html_data)
    fig = exp.as_pyplot_figure()
    fig.set_size_inches(24, 15, forward=True)
    pat = plt.savefig('./static/fig.png')

    return render_template('result.html', script=script, div=div, predictions=predict,html_data=html_data)


if __name__ == "__main__":
    app.run(debug=True)
