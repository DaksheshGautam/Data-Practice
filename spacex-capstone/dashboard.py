"""SpaceX Plotly Dash dashboard. Run python dashboard.py, then open localhost:8050."""
from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
df=pd.read_csv(Path(__file__).parent/'data/spacex_launch_dash.csv')
app=Dash(__name__)
app.layout=html.Div([
 html.H1('SpaceX Launch Records Dashboard'),
 html.P('Historical IBM course dataset. Class 1 means a successful first-stage landing.'),
 dcc.Dropdown(id='site-dropdown',options=[{'label':'All sites','value':'ALL'}]+[{'label':s,'value':s} for s in sorted(df['Launch Site'].unique())],value='ALL',clearable=False),
 dcc.Graph(id='success-pie-chart'),html.H3('Payload range (kg)'),
 dcc.RangeSlider(id='payload-slider',min=0,max=10000,step=500,value=[0,10000],marks={i:str(i) for i in range(0,10001,2000)},tooltip={'placement':'bottom','always_visible':True}),
 dcc.Graph(id='success-payload-scatter-chart'),html.Div(id='sample-count')
],style={'maxWidth':'1100px','margin':'30px auto','fontFamily':'Arial','color':'#152a3a'})

@app.callback(Output('success-pie-chart','figure'),Input('site-dropdown','value'))
def pie(site):
 if site=='ALL':
  q=df[df['class']==1].groupby('Launch Site').size().reset_index(name='Successful landings')
  return px.pie(q,names='Launch Site',values='Successful landings',title='Share of successful landings across launch sites')
 q=df[df['Launch Site']==site].groupby('class').size().reset_index(name='Launches'); q['Outcome']=q['class'].map({0:'Failure / no recovery',1:'Success'})
 return px.pie(q,names='Outcome',values='Launches',title=f'Landing outcomes: {site}')

@app.callback(Output('success-payload-scatter-chart','figure'),Output('sample-count','children'),Input('site-dropdown','value'),Input('payload-slider','value'))
def scatter(site,payload):
 q=df[df['Payload Mass (kg)'].between(*payload)]
 if site!='ALL':q=q[q['Launch Site']==site]
 fig=px.scatter(q,x='Payload Mass (kg)',y='class',color='Booster Version Category',hover_data=['Launch Site','Flight Number'],title='Payload and landing outcome')
 fig.update_yaxes(tickvals=[0,1],ticktext=['Failure / no recovery','Success'])
 return fig,f'{len(q)} launches in selected site and payload range'

if __name__=='__main__':app.run(debug=False,host='127.0.0.1',port=8050)
