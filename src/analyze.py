import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class Analyzer:

    template = 'seaborn'

    def __init__(self):
        pass

    def get_pie_chart(self, data, name, title):
        fig = px.pie(data, names=name, template=self.template, title=title)
        fig.update_traces(rotation=90, pull=[0.1], textinfo="percent+label")
        fig.show()

    def get_boxplot(self, data, name, target, title):
        fig = px.box(data, x=name, y=target, template=self.template, title=title)
        fig.show()

    def get_barplot(self, data, name, target, title, ylabel):
        axes1 = sns.barplot(data=data, x=name, y=target, errorbar=None)
        axes1.set_xlabel(name)
        axes1.set_ylabel(ylabel)
        axes1.set_ylim(0, 100)
        axes1.set_title(title)
        plt.show()

    def get_displot(self, data, name, stat, hue, height):
        sns.displot(data=data, x=name, stat=stat, hue=hue, height=height)
        plt.show()
