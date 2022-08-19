class Decision_Tree:
    def __init__(self,min_samples=3,max_depth=5):
        import numpy as np
        self.min_samples=min_samples
        self.max_depth=max_depth
        
    def check_unique(self,df):
        if len(np.unique(df[:,-1]))==1:
            return True
        return False
    
    def classify_model(self,df):
        label=np.unique(df[:,-1],return_counts=True)
        index= label[1].argmax()
        return label[0][index]
    
    def all_split_points(self,df):
        pot_split={}
        for i in range(df.shape[1]-1):
            col=i
            pot_split[col]=[]
            unique= np.unique(df[:,col])
            for val in range(len(unique)):
                if val!=0:
                    indi= unique[val]
                    kecmis= unique[val-1]
                    pot_split[col].append((indi+kecmis)/2)
        return pot_split
    
    def split_data(self,df,split_column,split_value):
        below= df[df[:,split_column]<=split_value]
        above= df[df[:,split_column]>split_value]
        return below,above
    
    def get_entropy(self,df):
        label=df[:,-1]
        _,counts= np.unique(label,return_counts=True)
        prob= counts/sum(counts)
        entropy= sum(prob*(-np.log2(prob)))
        return entropy
    
    
    def overall_entropy(self,below,above):
        l= len(below)+len(above)
        below_prob= len(below)/l
        above_prob= len(above)/l
        entropy= (below_prob*Decision_Tree.get_entropy(self,below)+above_prob*Decision_Tree.get_entropy(self,above))
        return entropy
    
    def get_best_split(self,df):
        entrop= 999999
        all_splits= Decision_Tree.all_split_points(self,df)
        for key,val in all_splits.items():
            for value in val:
                below,above= Decision_Tree.split_data(self,df,split_column=key,split_value=value)
                entr= Decision_Tree.overall_entropy(self,below,above)
                if entr<=entrop:
                    entrop=entr
                    best_column= key
                    best_point= value
        return best_column,best_point
    
    def decision_tree_classifier(self,df,counter=0):
        if counter==0:
            global l,cols
            l=[]
            cols=df.columns
            df=df.values
        if Decision_Tree.check_unique(self,df) or (len(df)<self.min_samples) or (counter==self.max_depth):
            classify= Decision_Tree.classify_model(self,df)
            return classify
        else:
            counter+=1
            col,value= Decision_Tree.get_best_split(self,df)
            l.append([col,value])
            below,above= Decision_Tree.split_data(self,df,split_column= col, split_value= value)

            q="{0}<={1}".format(cols[col],value)
            dec_tree= {q:[]}
            # 0 column 1 ise uygun deyer
            # Ex: sepal_width<=1.2


            yes_ans= Decision_Tree.decision_tree_classifier(self,below,counter=counter)
            no_ans= Decision_Tree.decision_tree_classifier(self,above,counter=counter)
            dec_tree[q].append(yes_ans)
            dec_tree[q].append(no_ans)
        return dec_tree
    def fit(self,df):
        self.tree= Decision_Tree.decision_tree_classifier(self,df)
        self.copy_tree=self.tree
        print("fitted")
    def predict(self,df):
        prediction=[]
        def inner(self,df):
            q=list(self.copy_tree.keys())[0]
            feat,value= q.split('<=')
            if df[feat]<=float(value):
                self.copy_tree= self.copy_tree[q][0]
            else:
                self.copy_tree= self.copy_tree[q][1]

            if  type(self.copy_tree)==dict:
                return inner(self,df)
            else:
                answer=self.copy_tree
                self.copy_tree=self.tree
                return answer
        print(type(df))
        for _,row in df.iterrows():
            #print(row['petal_width'])
            prediction.append(inner(self,row))
        return prediction
