"""Interactive historical churn explorer for retention analysts."""
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data import load_data, OUTPUT
from src.statistics import segment_summary, wilson_interval
from src.reports import load_model_reports

logger = logging.getLogger(__name__)

st.set_page_config(page_title='Credit Card | Retention Insights', page_icon='📊',layout='wide')
COLORS = {'Existing Customer':'#0072B2','Attrited Customer':'#D55E00'}

@st.cache_data
def customers():
    return load_data()

@st.cache_data
def model_reports(results_modified, importance_modified):
    return load_model_reports(OUTPUT)

st.title('Credit card retention insights')
st.write('Explore recorded attrition, compare customer segments and review a classification prototype.')
st.caption('Historical snapshot · 10,127 customers · Educational analysis, not a live bank monitoring system')
try:
    df = customers()
except (OSError, ValueError, pd.errors.ParserError):
    logger.exception('Customer snapshot failed validation or could not be loaded')
    st.error('The customer data is temporarily unavailable. Please contact the project maintainer.')
    st.stop()

def reset_filters():
    # Callbacks run before widgets are reconstructed, restoring their explicit state.
    st.session_state['cards'] = sorted(df.Card_Category.unique())
    st.session_state['products'] = (1, 6)
    st.session_state['transactions'] = (int(df.Total_Trans_Ct.min()), int(df.Total_Trans_Ct.max()))
    st.session_state['include_missing'] = True

with st.sidebar:
    st.header('Explore a segment')
    st.caption('Filters apply to the overview and activity views. Model evaluation always uses the fixed test split.')
    cards = st.multiselect('Card category',sorted(df.Card_Category.unique()),default=sorted(df.Card_Category.unique()),key='cards')
    products = st.slider('Products held',1,6,(1,6),key='products')
    transactions = st.slider('Annual transactions',int(df.Total_Trans_Ct.min()),int(df.Total_Trans_Ct.max()),
                             (int(df.Total_Trans_Ct.min()),int(df.Total_Trans_Ct.max())),key='transactions')
    include_missing = st.checkbox('Include customers with unrecorded demographics',value=True,key='include_missing')
    st.caption('Use Reset filters to restore the full snapshot.')
    st.button('Reset filters',on_click=reset_filters)
selected = df[df.Card_Category.isin(cards) & df.Total_Relationship_Count.between(*products)
              & df.Total_Trans_Ct.between(*transactions)].copy()
if not include_missing:
    selected = selected.dropna(subset=['Income_Category','Education_Level','Marital_Status'])
overview,activity,model_tab,about = st.tabs(['Portfolio overview','Activity & relationships','Model review','About the data'])
with overview:
    st.subheader('How much attrition does this segment show?')
    if selected.empty:
        st.info('No customers match these filters. Broaden the ranges or select a card category.')
    else:
        count = len(selected)
        churned = int(selected.Churn.sum())
        low,high = wilson_interval(churned,count)
        a,b,c,d = st.columns(4)
        a.metric('Customers',f'{count:,}')
        b.metric('Recorded attrition',f'{churned/count:.1%}')
        c.metric('Attrited customers',f'{churned:,}')
        d.metric('Median annual transactions',f'{selected.Total_Trans_Ct.median():.0f}')
        st.caption(f'Attrition estimate: {low:.1%}–{high:.1%} (95% Wilson interval); snapshot-wide rate {df.Churn.mean():.1%}.')
        if count < 30:
            st.warning('Fewer than 30 customers: this segment is too small for a stable rate comparison.')
        left,right = st.columns(2)
        with left:
            counts = selected.Attrition_Flag.value_counts().rename_axis('Status').reset_index(name='Customers')
            fig = px.bar(counts,x='Status',y='Customers',color='Status',color_discrete_map=COLORS,
                         title='Customer outcomes in the selected segment',text='Customers')
            fig.update_layout(showlegend=False,xaxis_title=None)
            st.plotly_chart(fig,width='stretch')
        with right:
            rates = segment_summary(selected,'Total_Relationship_Count')
            rates['Attrition (%)'] = rates.churn_rate*100
            fig = px.bar(rates,x='Total_Relationship_Count',y='Attrition (%)',
                         hover_data=['customers','churned'],title='Attrition by products held',
                         labels={'Total_Relationship_Count':'Products held'},color_discrete_sequence=['#0072B2'])
            fig.update_traces(error_y=dict(type='data',array=(rates.ci_upper-rates.churn_rate)*100,
                                           arrayminus=(rates.churn_rate-rates.ci_lower)*100))
            st.plotly_chart(fig,width='stretch')
        st.write('In the full snapshot, smaller product portfolios show greater historical attrition; filtered segments can differ. This association does not establish that adding a product will prevent departure.')
        rates_display = rates[['Total_Relationship_Count','customers','churned']].rename(
            columns={'Total_Relationship_Count':'Products held','customers':'Customers','churned':'Attrited customers'})
        rates_display['Attrition (%)'] = (rates.churn_rate*100).round(2)
        rates_display['95% lower (%)'] = (rates.ci_lower*100).round(2)
        rates_display['95% upper (%)'] = (rates.ci_upper*100).round(2)
        st.dataframe(rates_display,hide_index=True,width='stretch')
        # Export aggregates only; avoid offering individual customer records in the dashboard.
        st.download_button('Download segment summary (CSV)',rates_display.to_csv(index=False),
                            file_name='segment_summary.csv',mime='text/csv')
with activity:
    st.subheader('Which behaviours accompany recorded attrition?')
    if selected.empty:
        st.info('No customers match the current filters.')
    else:
        fig = px.histogram(selected,x='Total_Trans_Ct',color='Attrition_Flag',histnorm='percent',
                           barmode='overlay',opacity=0.65,nbins=30,color_discrete_map=COLORS,
                           labels={'Total_Trans_Ct':'Annual transaction count','Attrition_Flag':'Recorded status'},
                           title='Transaction frequency, normalised within each status')
        st.plotly_chart(fig,width='stretch')
        st.caption('Each status distribution totals 100%; this compares distribution shapes rather than class sizes.')
        with st.expander('Transaction frequency data table'):
            frequency = selected.groupby(['Attrition_Flag','Total_Trans_Ct']).size().reset_index(name='Customers')
            frequency['Share of status group (%)'] = frequency.Customers / frequency.groupby('Attrition_Flag').Customers.transform('sum') * 100
            st.dataframe(frequency.rename(columns={'Attrition_Flag':'Recorded status','Total_Trans_Ct':'Annual transactions'}),hide_index=True,width='stretch')

        chart_data = selected.assign(Status=selected.Attrition_Flag)
        fig = px.density_heatmap(chart_data,x='Total_Trans_Ct',y='Total_Revolving_Bal',facet_col='Status',
                                nbinsx=25,nbinsy=20,color_continuous_scale='Blues',
                                labels={'Total_Trans_Ct':'Annual transaction count','Total_Revolving_Bal':'Revolving balance ($)'},
                                title='Where customers concentrate: activity and balance')
        st.plotly_chart(fig,width='stretch')
        st.caption('Bin colours show customer counts, not churn rates. The retained class is much larger; compare locations within each panel.')
        contact = segment_summary(selected,'Contacts_Count_12_mon')
        contact['Attrition (%)'] = contact.churn_rate*100
        fig = px.line(contact,x='Contacts_Count_12_mon',y='Attrition (%)',markers=True,
                      hover_data=['customers'],title='Contact frequency and recorded attrition',
                      labels={'Contacts_Count_12_mon':'Annual contacts'},color_discrete_sequence=['#D55E00'])
        st.plotly_chart(fig,width='stretch')
        with st.expander('Contact frequency data table'):
            contact_table = contact.rename(columns={'Contacts_Count_12_mon':'Annual contacts','customers':'Customers',
                                                   'churned':'Attrited customers','ci_lower':'95% lower (%)','ci_upper':'95% upper (%)'})
            contact_table[['95% lower (%)','95% upper (%)']] *= 100
            st.dataframe(contact_table.drop(columns='churn_rate').round(2),hide_index=True,width='stretch')
        st.write('The full snapshot shows associations between contact, activity and attrition; filtered segments can differ. Contacts may include closure administration; there is no timestamp to establish which happened first.')
with model_tab:
    st.subheader('Can account activity distinguish the recorded outcomes?')
    report_path = OUTPUT/'model_results.json'
    importance_path = OUTPUT/'validation_importance.csv'
    report = None
    try:
        report, importance = model_reports(report_path.stat().st_mtime_ns,importance_path.stat().st_mtime_ns)
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        logger.exception('Saved model evidence failed validation or could not be loaded')
        st.info('Model review is temporarily unavailable. Portfolio and activity exploration remain available.')
    if report is not None:
        result = report['test']
        st.write(f"Selected model: **{report['selected_model']}**. Selection used training cross-validation; the decision threshold used validation data.")
        st.caption('Sidebar filters do not change these fixed test results.')
        a,b,c = st.columns(3)
        a.metric('Churn cases detected (recall)',f"{result['recall']:.1%}")
        b.metric('Flagged cases that churned (precision)',f"{result['precision']:.1%}")
        c.metric('Average precision',f"{result['average_precision']:.3f}")
        st.caption(f"Test set: {report['split_sizes']['test']:,} customers. Majority baseline average precision: {report['dummy_test']['average_precision']:.3f}.")
        matrix = [[result['tn'],result['fp']],[result['fn'],result['tp']]]
        fig = go.Figure(go.Heatmap(z=matrix,x=['Predicted retained','Predicted attrited'],
                                  y=['Actually retained','Actually attrited'],text=matrix,texttemplate='%{text}',
                                  colorscale='Blues',showscale=False))
        fig.update_layout(title='Classification outcomes on the internal test split',height=350)
        st.plotly_chart(fig,width='stretch')
        with st.expander('Classification outcomes data table'):
            st.dataframe(pd.DataFrame(matrix,index=['Actually retained','Actually attrited'],
                                      columns=['Predicted retained','Predicted attrited']),width='stretch')
        st.write(f"The prototype missed {result['fn']} attrited customers and flagged {result['fp']} retained customers. A bank would need to agree the costs of these errors before choosing a production threshold.")
        feature_labels = {
            'Total_Relationship_Count':'Products held','Months_Inactive_12_mon':'Inactive months',
            'Contacts_Count_12_mon':'Annual contacts','Credit_Limit':'Credit limit',
            'Total_Revolving_Bal':'Revolving balance','Total_Amt_Chng_Q4_Q1':'Transaction amount change',
            'Total_Trans_Amt':'Annual transaction amount','Total_Trans_Ct':'Annual transaction count',
            'Total_Ct_Chng_Q4_Q1':'Transaction count change','Avg_Utilization_Ratio':'Credit utilisation'}
        importance['feature'] = importance.feature.map(feature_labels).fillna(importance.feature)
        st.plotly_chart(px.bar(importance.sort_values('ap_decrease'),x='ap_decrease',y='feature',orientation='h',
                              labels={'ap_decrease':'Drop in validation average precision','feature':'Activity measure'},
                              title='Which features does the model rely on?'),width='stretch')
        with st.expander('Feature reliance data table'):
            st.dataframe(importance.rename(columns={'feature':'Activity measure','ap_decrease':'Average precision decrease',
                                                   'sd':'Variation across permutations'}),hide_index=True,width='stretch')
        st.caption('Permutation importance measures reliance, not causation. Correlated features can share importance.')
        st.warning('This is an internal snapshot classification experiment. Unit 1 explored this dataset already. Independent, dated customer data is needed before claiming future churn prediction or deploying customer decisions.')
with about:
    st.subheader('Source, scope and decisions')
    st.caption('Mentor portfolio adaptation maintained by Sanif Mujawar. Based on Muhammad Asim\'s credited Unit 2 project, with OpenAI Codex development assistance.')
    st.write('Credit Card customers dataset, published by Sakshi Goyal on Kaggle; original source attribution: Analyttica LEAPS. One row represents a customer snapshot.')
    st.link_button('Dataset source','https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers')
    st.write('Customer identifiers, stored classifier outputs and the redundant open-to-buy field are excluded. Unknown demographic values remain missing; no customer rows are dropped in the default view.')
    st.write('The model uses ten numerical account activity features. Demographics, card tier and tenure are excluded. This does not establish fairness: account behaviour can still contain proxies and requires review.')
    st.write('No dated observation window, intervention outcome, retention cost or customer value is supplied. Recommendations are hypotheses for future controlled testing.')
    st.caption('Product rates, transaction frequency, contact rates, classification outcomes and feature reliance have data-table alternatives. Plotly supports hover detail and zoom; labels identify status as well as colour.')
