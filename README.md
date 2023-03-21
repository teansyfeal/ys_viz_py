# Data analysis and visualization with Pandas, MatPlotLib and Seaborn packages


## Stacked-Area plot

    import numpy as np
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import stats

    plt.rcParams.update({"font.family": "Reem Kufi"})
    df = pd.read_csv('vg_trend.csv')
    df['Year_of_Release'] = pd.to_datetime(df['Year_of_Release'], format='%Y')
    stream = df[['Year_of_Release',
                 'NA_Sales',
                 'EU_Sales',
                 'JP_Sales',
                 'Other_Sales']].resample('1Y', on = 'Year_of_Release').sum()
    ax = stream.plot(kind='area', stacked=True)
    ax.legend(loc = 'upper left')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sales')

![image](https://user-images.githubusercontent.com/58554631/226743593-f797e214-4b92-47af-b87a-47f1d0134bf3.png)

## Stream Graph

    fig, ax = plt.subplots(figsize=(10,5))

    # pal = sns.color_palette("Set3")
    pal = sns.color_palette("terrain_r", 10)

    ax.stackplot(stream.index,
                 stream.T,
                 baseline='wiggle',
                 colors = pal,
                 labels = stream.columns)
    ax.legend(loc = 'upper left')
    ax.set_facecolor('lightgrey')
    plt.title("Stream Graph of sales",
              fontsize = 15,
              backgroundcolor = 'grey',
              color = 'white')

![image](https://user-images.githubusercontent.com/58554631/226743752-246ae7f1-1493-4e94-8872-3a5f83b1c55c.png)

## Heatmap

    df = pd.read_csv('vg_trend.csv')
    df = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    df1 = df[df['Year_of_Release'] >= 1992]
    df1 = df1[df1['Year_of_Release'] <2017]

    heatmap = pd.pivot_table(df1, 
                             values = 'Global_Sales',
                             index=['Genre'],
                             columns='Year_of_Release')


    fig, ax = plt.subplots(figsize=(16,8))

    sns.set()
    sns.heatmap(heatmap, cbar=True)
    plt.title('Heatmap', size = 20)
    plt.xlabel('')
    plt.xticks(rotation=60)
    plt.show()

![image](https://user-images.githubusercontent.com/58554631/226743988-4bf57918-6d7d-484c-9fdc-51603c194fb5.png)

## Word Cloud

    import pandas as pd
    import matplotlib.pyplot as plt
    %matplotlib inline
    from wordcloud import WordCloud
    
    df = pd.read_csv('anime.csv')
    new_df = df[df['episodes'].notnull()]
    new_df = new_df[ df['score'] >= 8.5]
    text = " ".join(title for title in new_df.title).lower()
    word_cloud = WordCloud(collocations = True, background_color = 'white').generate(text)

    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

![image](https://user-images.githubusercontent.com/58554631/226744066-d6761ed9-09dc-431c-a716-f6da75305857.png)

## Area plots

    import numpy as np
    import seaborn as sns
    import pandas as pd

    df = pd.read_csv('vg_trend.csv')
    df['Year_of_Release'] = pd.to_datetime(df['Year_of_Release'], format='%Y')

    new_df = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()

    grid = sns.FacetGrid(new_df, col='Genre', hue='Genre', col_wrap=4)
    grid = grid.map(plt.plot, 'Year_of_Release', 'Global_Sales')
    grid = grid.map(plt.fill_between, 'Year_of_Release', 'Global_Sales', alpha=0.2).set_titles("{col_name} Genre")
    grid = grid.set_titles("{col_name}")
    plt.subplots_adjust(top=0.92)
    grid = grid.fig.suptitle('Global sales of different videogame genres')
    plt.show()

![image](https://user-images.githubusercontent.com/58554631/226744153-dea7873a-3046-4e49-aa30-b92afd414dc1.png)

## Dendrogram

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline 
    df = pd.read_csv('vg_trend.csv')
    df.head()
    df = df.sort_values('User_Count',
                        ascending=False).groupby('Developer').head(20)
    data = df[df['Developer'].notnull()]
    dendr = pd.DataFrame(data[['NA_Sales',
                 'EU_Sales',
                 'JP_Sales',
                 'Other_Sales']])
    from scipy.cluster.vq import whiten
    scaled_data = whiten(dendr.to_numpy())
    from scipy.cluster.hierarchy import fcluster, linkage
    distance_matrix = linkage(scaled_data, method = 'ward', metric = 'euclidean')

    from scipy.cluster.hierarchy import dendrogram
    dn = dendrogram(distance_matrix)
    plt.show()

![image](https://user-images.githubusercontent.com/58554631/226744423-8e34cb13-b0d8-4a56-9bd2-1005b460e273.png)
