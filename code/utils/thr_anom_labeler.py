
class ThresholdAnomalyLabeler:
    """
        dataframe -- pandas dataframe
        logop     -- logical operator in {"and", "or"}
        columns   -- columns that have thresholds
        threshold -- threshold values
        relop     -- list of threshold relational operators in {"<", ">"}
    """
    def __init__(self, dataframe, logop, columns, thresholds, relop, ignore_zero_values=False):
        self.df = dataframe
        
        if logop != 'and' and logop != 'or':
            raise Exception("[ThresholdAnomalyLabeler] Invalid logop " + logop)
           
        self.logop = logop   
           
        identical_dimensions = len(columns) == len(thresholds) == len(relop)            
        if not identical_dimensions:
            raise Exception("[ThresholdAnomalyLabeler] Columns, thresholds, and relop should be the same dimension")
        
        for rop in relop:
            if rop != '<' and rop != '>':
                raise Exception("[ThresholdAnomalyLabeler] Invalid relational operator" + rop)
            
        self.columns = columns
        self.thresholds = thresholds
        self.relop = relop
        
        self.ignore_zero_values = ignore_zero_values
        
        self.identify_anomalies()
        print("Total instances", self.df.shape[0])
        print("Total anomalies", self.num_anomalies)
        
        
    def identify_anomalies(self):
        self.num_anomalies = 0
        relevant_columns = self.df[self.columns] 
        num_columns = len(self.columns)
        self.anomaly_vector = []
        
        for _, row in relevant_columns.iterrows():
            lvalues = []
            for i in range(num_columns):
                val = row[self.columns[i]]
                if self.ignore_zero_values and val == 0:
                    continue
                
                rel = self.relop[i]
                thr = self.thresholds[i]
                expr = str(val) + rel + str(thr)
                res = eval(expr)
                lvalues.append(res)
                
            if self.logop == 'and':
                is_anomaly = all(lvalues)
            else:
                is_anomaly = any(lvalues)
                
            if is_anomaly:
                self.num_anomalies += 1
                
            self.anomaly_vector.append("yes" if is_anomaly else "no")
            
            
    def save(self, outfile):
        self.df['ANOMALY'] = self.anomaly_vector
        self.df.to_csv(outfile, index=False)
        return self.df

    