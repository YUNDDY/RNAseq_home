from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse

from uploads.core.models import Document
from uploads.core.forms import DocumentForm

import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import base64
import tempfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import venn
import itertools
import numpy as np

def home(request):
    documents = Document.objects.all()
    return render(request, 'core/home.html', { 'documents': documents })


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        cluster_r = request.POST.get('cluster_r')
        cluster_r = True if cluster_r == 'on' else False
        cluster_c = request.POST.get('cluster_c')
        cluster_c = True if cluster_c == 'on' else False
        gene_view = request.POST.get('gene_view')
        gene_view = True if gene_view == 'on' else False
        color = request.POST.get('color')

        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content_10 = pd.read_excel(f).head(10)
            f.seek(0)  # 파일 포인터를 파일의 시작으로 되돌립니다.
            df = pd.read_excel(f, index_col=0)
        rows = df.shape[0]  # 행의 개수
        cols = df.shape[1]  # 열의 개수
        fig_width = cols * 1  # 열의 개수에 따라 가로 크기 계산
        fig_height = rows * 0.27  # 행의 개수에 따라 세로 크기 계산

        cbar_pos = [0.02, 0.87, 0.05, 0.1]

        if cluster_c and cluster_r:
            g = sns.clustermap(df, cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=True, row_cluster=True, cbar_pos=cbar_pos, dendrogram_ratio=0.15)
        elif cluster_c:
            g = sns.clustermap(df, cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=True, row_cluster=False, cbar_pos=cbar_pos, dendrogram_ratio=0.15)
        elif cluster_r:
            g = sns.clustermap(df, cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=False, row_cluster=True, cbar_pos=cbar_pos, dendrogram_ratio=0.15)
        else:
            g = sns.clustermap(df, cmap=color, square=True, annot=False, figsize=(fig_width, fig_height), col_cluster=False, row_cluster=False, cbar_pos=cbar_pos, dendrogram_ratio=0.15)
        #g.fig.suptitle('Heatmap', fontsize=16)

        if gene_view:
            plt.setp(g.ax_heatmap.get_yticklabels(), visible=False)
        else:
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0) # Rotate y-axis labels for better visibility
    
    # Encode the heatmap image in base64 format
        heatmap_buffer = io.BytesIO()
        plt.savefig(heatmap_buffer, format='png')
        heatmap_image = heatmap_buffer.getvalue()
        heatmap_buffer.close()
        heatmap_image_base64 = base64.b64encode(heatmap_image).decode('utf-8')
        heatmap_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h3>Heatmap</h3><img src="data:image/png;base64,{heatmap_image_base64}"/></div>'    

        heatmap_download_link = f'data:image/png;base64,{heatmap_image_base64}'

        return render(request, 'core/heatmap.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content_10,
            'heatmap_html': heatmap_html,
            'heatmap_download_link': heatmap_download_link,
        })
    return render(request, 'core/heatmap.html')



def Venn(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']

        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content_10 = pd.read_excel(f).head(10)
            f.seek(0)  # 파일 포인터를 파일의 시작으로 되돌립니다.
            df = pd.read_excel(f).fillna('')

        sets = [set(filter(None, df.iloc[:, i])) for i in range(df.shape[1])]
        labels = venn.get_labels(sets, fill=['number'])
        output = []
        gene_lists = {}
        set_labels = {}
        n = len(list(labels.keys())[0]) - 1  # Get the length of the label combinations
        for label, gene_indices in labels.items():
            set_indices = tuple(int(bit) for bit in label)
            gene_lists[label] = set()
            loc=''
            loc1=''
            loc2=''
            SETS=''
            for i, bit in enumerate(set_indices):
                if bit == 1:
                    SETS += df.columns.values[i] +"&"
                    loc1 += "sets[" + str(i) + "]&"
            loc1 = loc1.rstrip('&')
            SETS= "only_" + SETS.rstrip('&')
            for i, bit in enumerate(set_indices):
                if bit == 0:
                    loc2 += "-sets[" + str(i) + "]"
            loc = loc1 + loc2
            output.append([SETS] + list(eval(loc)))

            output_df = pd.DataFrame(output).T

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            output_df.to_csv(temp.name, index=False,header=False)
            download_link = temp.name

        venn.venn3(labels, names=df.columns.tolist())

        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format='png')
        image_buffer.seek(0)
        image_base64 = base64.b64encode(image_buffer.read()).decode('utf-8')
        image_html = f'<img src="data:image/png;base64,{image_base64}"/>'
        venn_download_link = f'data:image/png;base64,{image_base64}'
        

        return render(request, 'core/venn.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content_10,
            'plot_venn' : image_html,
            'download_link': download_link,
            'venn_download_link': venn_download_link,

        })
    return render(request, 'core/venn.html')

def PCA_plot(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content_10 = pd.read_excel(f).head(10)
            f.seek(0)  # 파일 포인터를 파일의 시작으로 되돌립니다.
            df = pd.read_excel(f, index_col=0)
        
        # Remove the second row (group information)
        groups = df.iloc[0]
        df = df.iloc[1:]

        # Transpose the dataframe
        df = df.transpose()

        # Perform PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(df.values)
        
        # Get the group information
        # groups = df.iloc[:, 0].values
        
        # Create a scatter plot with different colors for each group
        fig, ax = plt.subplots(figsize=(8, 8))
        for group in np.unique(groups):
            indices = np.where(groups == group)
            ax.scatter(pca_results[indices, 0], pca_results[indices, 1], label=group)
        ax.legend()

        arrowprops = {
            'arrowstyle': '-'
            }

        for i, sample_id in enumerate(df.index):
            #ax.text(pca_results[i, 0], pca_results[i, 1], sample_id, ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.04'))
            ax.annotate(text=sample_id, xy=(pca_results[i, 0], pca_results[i, 1]), xytext=(pca_results[i, 0]-1, pca_results[i, 1]+5), arrowprops=arrowprops, fontsize=10)

        # Save the scatter plot as a PNG image
        scatter_buffer = io.BytesIO()
        plt.savefig(scatter_buffer, format='png')
        scatter_image = scatter_buffer.getvalue()
        scatter_buffer.close()

        # Encode the PNG image as base64
        scatter_image_base64 = base64.b64encode(scatter_image).decode('utf-8')

        scatter_plot_html = f'<div style="text-align: left;margin: 0 0 0 0;"><h3>Graph</h3><img src="data:image/png;base64,{scatter_image_base64}"/></div>'
        pca_download_link = f'data:image/png;base64,{scatter_image_base64}'

        return render(request, 'core/PCA.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content_10,
            'plot_pca': scatter_plot_html,
            'pca_download_link': pca_download_link,
        })

    return render(request, 'core/PCA.html')




def data_filter(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)

        with fs.open(filename) as f:
            file_content = f.read().decode()

        file_content_10 = pd.read_csv(io.StringIO(file_content), sep='\t').head(10)
        data = io.StringIO(file_content)
        df = pd.read_csv(data, sep='\t', index_col=0)

        if request.POST.get('logFC_threshold') and request.POST.get('pval_threshold') and request.POST.get('FDR_threshold'):
            logFC_threshold = float(request.POST.get('logFC_threshold'))
            pval_threshold = float(request.POST.get('pval_threshold'))
            FDR_threshold = float(request.POST.get('FDR_threshold'))
        else:
            logFC_threshold = 1.0
            pval_threshold = 0.05
            FDR_threshold = 0.05

        filtered_df = df[(df['logFC'] > logFC_threshold) & (df['FDR'] < FDR_threshold)]
        filtered_df_10 = filtered_df.head(10)

        with tempfile.NamedTemporaryFile(delete=False) as temp:
            filtered_df.to_csv(temp.name, index=False)

        download_link = temp.name


        return render(request, 'core/filtering.html', {
            'uploaded_file_url': uploaded_file_url,
            'file_content': file_content_10,
            'filtered_df' : filtered_df_10,
            'download_link': download_link,
            'logFC_threshold': logFC_threshold,
            'FDR_threshold': FDR_threshold,
        })

    return render(request, 'core/filtering.html')


def download_filtered(request):
    filepath = request.GET.get('file')
    with open(filepath, 'rb') as f:
        response = HttpResponse(f, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="filtered_data.csv"'
        return response


def model_form_upload(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'core/model_form_upload.html', {
        'form': form
    })
