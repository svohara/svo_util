'''
Created on Dec 10, 2012
@author: Stephen O'Hara

Utility functions for interacting with certain types
of data files.
'''
import os
import numpy as np
import svo_util
from StringIO import StringIO
import tokenize

find = np.nonzero  #this is the matlab command
which = np.nonzero #this is the R command

def smart_split(linein, sep=","):
    '''
    Works much like built-in split method of strings, but this version
    will honor quoted strings. Meaning that if the separator is inside a quoted string,
    then the separator is ignored in favor of keeping quoted strings as a single token.
    '''
    curtokens = []
    fieldvals = []
    prev_end = 0
    tkz = tokenize.generate_tokens(StringIO(linein).readline)
    for _, tokval, (_,colstart), (_,colend), _ in tkz:
        ws_delta = colstart - prev_end
        prev_end = colend
        
        if tokval.strip() == '': continue  #ignore whitespace
        
        if ''.join(tokval.split(sep)) == '':
            if len(curtokens) > 0:
                fieldvals.append( ("".join(curtokens)).strip() )
                curtokens = []
                continue
            else:
                continue
        
        if ws_delta > 0:
            tokval = (" "*ws_delta)+tokval
        
        if (tokval[0] == tokval[-1] == '"') or (tokval[0] == tokval[-1] == '\''):
            tokval = tokval[1:-1]
            
        curtokens.append(tokval)
    #at end of line, we will likely have curtokens and no separator encountered
    if len(curtokens) > 0: fieldvals.append( ("".join(curtokens)).strip() )
        
    return fieldvals

class CSV_data:
    '''
    Class to encapsulate reading data from simple csv files
    where the first row is the column headers and fields are
    separated with commas. Rows are data samples, columns are field values.
    '''
    def __init__(self, datadir, csvfile, label_cols=[0], separator=",", missing_val_sentinel=-9999):
        '''
        constructor
        @param datadir: The directory where the csv file is located
        @param csvfile: The filename of the data file
        @param label_cols: A list of the zero-indexed column numbers
        that should be treated as labels. These columns are assumed
        to have a discrete set of values, like a set of string labels,
        a set of integers. For regression data sets where the "labels"
        are continuous-valued numbers, set label_cols = None, and all
        the csv fields will be loaded into the data matrix.
        '''
        self.datadir = datadir
        self.csvfile = csvfile
        self.label_cols = label_cols
        self.label_dict = {}  # key is label_col #, value is a list of the values of each sample
        self.label_names = [] #list of the field names of the label columns
        self.fields = [] #list of the field names for the data columns (non-labels)
        self.separator = separator
        self.skipped_rows = []  #row skipped if a data field was empty
        self.missing_val_sentinel = missing_val_sentinel
        self._load_data()
     
    def _parse_row(self, linedat):
        linedat = linedat.strip()
        tmpfields = [ fld.strip() for fld in linedat.split(self.separator)]
        
        #are any field values empty? then badrow if missing_val_sentinel is none
        if self.missing_val_sentinel is None:  
            if '' in tmpfields: return None
        
        #handle label columns
        for col in self.label_cols:
            fieldval = tmpfields[col]
            self.label_dict[col].append( fieldval )
            
        #non-label columns are converted to floats, appended into an ordered list
        #replace any non-numeric entries in tmpfields with the sentinel number
        data_cols = sorted( set(range(len(tmpfields)))-set(self.label_cols) )
        data_fields = [ tmpfields[i] for i in data_cols]
        row_dat = [ svo_util.parse_number(s, fail=self.missing_val_sentinel) for s in data_fields ]
            
        return row_dat
        
    def _load_data(self):
        '''
        internal function that reads the csv text data and converts
        the values as appropriate to the internal data representation
        '''
        infile = os.path.join(self.datadir,self.csvfile)
        with open(infile,"r") as f:
            lines = f.readlines()
        first_row = self._load_header(lines)
        raw_dat = lines[(first_row+1):]  #drop the header row and any leading blank rows
        
        data_list = []   
        for i,linedat in enumerate(raw_dat):
            row_num = i + first_row + 1 #original row from csv file including header/initial blank lines
            row_dat = self._parse_row(linedat)
            if row_dat is None:
                self.skipped_rows.append(row_num)
                print "Warning: row %d of input skipped with missing values."%row_num
                #print "Line data: %s"%str(linedat)
            else:
                data_list.append(row_dat)
         
        self.data = np.array(data_list)
        print "Loaded data matrix of size: %d by %d"%self.data.shape
        print "The header row was line %d"%(first_row+1) #one-based for user
        print "The label columns are: %s"%str(self.label_names)
        
    def _load_header(self, lines):
        '''
        header has the field names, and should
        be the first non-empty line of the csv file
        '''
        i = 0
        while lines[i].strip() == '':
            i+=1
        first_row = i
        self.fields = [ fld.strip() for fld in lines[first_row].split(self.separator) ] 
        
        #setup initial label_dict entries for label fields
        for col in self.label_cols:
            self.label_dict[ col ] = []
            self.label_names.append(self.fields[col])
            
        #self.fields will have the field names for the data (non-label) columns in order
        for label_name in self.label_names:
            self.fields.remove(label_name)
                
        return first_row
    


class C45_data:
    '''
    Class to encapsulate functions to read data from C4.5 formatted
    data mining files. Characterized by text files with .names and .data
    extensions.
    '''
    def __init__(self, datadir, namefile, datafile, missing_val_sentinel=-9999):
        '''
        Constructor
        @param datadir: The directory where the data files are located
        @param namefile: The filename that has the .names information. You may
        specify as xyz.names or just xyz (with .names extension inferred)
        @param datafile: The filename that has the .data information. You may
        specify as xyz.data or just xyz (with .data extension inferred)
        @param missing_val_sentinel: Set to a numeric value that can uniquely identify
        fields in the data matrix where the values were missing. Matrix can be
        transformed via functions like 'replace_missing_values_with_col_means()'. Set
        to None to have an error thrown if any missing values are encountered.
        @note: Do NOT include the path in the namefile or datafile parameters. These
        are assumed to be files in the datadir specified.
        '''
        self.classes = []
        self.fields = []
        self.fieldtypes = []
        self.data = []
        self.labels = []
        
        self.datadir = datadir
        self.namefile = namefile if namefile[-5:] == "names" else "%s.names"%namefile
        self.datafile = datafile if datafile[-4:] == "data" else "%s.data"%datafile
        
        self.missing_val_sentinel = missing_val_sentinel
        
        self._loadNames()
        self._loadData()
        
    def _loadNames(self):
        nfile = os.path.join(self.datadir, self.namefile)
        assert os.path.exists(nfile)
        
        self.classes = []
        self.fields = []
        self.fieldtypes = []
        
        with open(nfile,"r") as f:
            lines = f.readlines()
            
        #first line is supposed to be the class names
        tmp = lines[0].split(",")
        self.classes = [c.strip() for c in tmp]
        print "There are %d classes defined by data set: %s"%(len(self.classes), str(self.classes))
        
        for tmp2 in lines[1:]:
            #if the line is just whitespace, skip it
            if tmp2.strip() == '': continue
            tmp3 = tmp2.split(':')
            self.fields.append( tmp3[0].strip())
            self.fieldtypes.append( tmp3[1].strip() )
        
        print "There are %d fields in the data set."%len(self.fields)
        
    def _parse_row(self, tmpfields):
        '''
        Given a list of strings representing one row of data, return a vector of numbers.
        This will replace missing values with the sentinel number, or raise an error if
        self.missing_val_sentinel is None.
        '''        
        #are any field values 'empty' or non-numeric
        try:
            V = np.array(tmpfields, dtype=float)
        except(ValueError):
            #There are values in tmpfields that can't be converted
            # to floats, we assume these are missing values.
            if self.missing_val_sentinel is None:
                raise(ValueError)
            else:
                #replace any non-numeric entries in tmpfields with the sentinel number
                tmp2 = [ svo_util.parse_number(s, fail=self.missing_val_sentinel) for s in tmpfields ]
                V = np.array(tmp2, dtype=float)
                    
        return V
       
    def _loadData(self):
        '''
        Loads the data. self.data will be a numpy matrix representing the source data.
        Each row is a sample, and each column is a feature.
        '''
        dfile = os.path.join(self.datadir, self.datafile)
        assert os.path.exists(dfile)
        data = []
        labels = []
        
        with open(dfile, "r") as f:
            datalines = f.readlines()
            
        for dl in datalines:
            tmp = dl.split(',')
            V = self._parse_row( tmp[:-1] )
            data.append( V )
            labels.append(tmp[-1].strip())  #last entry for each row is the label
            
        print "Loaded %d lines of data."%len(data)
        self.data = np.array(data, dtype=float)
        self.labels = np.array(labels)
        P = self.data.shape[1]
            
        #some of the data features may be constant (stddev=0), so they need to be dropped
        cc = svo_util.constant_cols(self.data)
        if len(cc) > 1:
            print "Warning, %d features in data set have zero standard deviation."%len(cc)
            print "Those features will be dropped."
            all_cols = set(range(P))
            good_cols = sorted( list(all_cols - set(cc))) #set difference
            #filter field names to remove bad columns
            self.fields = [ f for i,f in enumerate(self.fields) if i in good_cols]
            self.fieldtypes = [ f for i,f in enumerate(self.fieldtypes) if i in good_cols]
            #filter data array to remove bad columns
            mask = np.array(good_cols)
            self.data = self.data[:,mask]
            #remember which were the bad columns as an object property
            self.constant_cols = cc
    
    def labelsAsIntegers(self, include_label_set=False):
        '''
        Returns the labels as a list of integer values, and the accompanying ordered
        list that tells you which text label is represented by which integer
        @param include_label_set: If True, then the return will be a tuple of
        the label integers and a list of the unique labels in the index order
        of the integers. If False, just the label integers are provided in a single
        value return.
        @return: either int_list or tuple (int_list, unique_labels)
        '''
        unique_labels = sorted(list(set(self.labels)))
        integer_labels = [ unique_labels.index(lbl) for lbl in self.labels]
        if include_label_set:
            return (integer_labels , unique_labels)
        else:
            return integer_labels
        
    def labelsAsIndicatorVals(self, include_label_set=False):
        '''
        Returns the labels as indicator variables. Indicator variables have a column for
        each possible class, and either a 0 or 1 in that column if that label applies.
        It is common for classification algorithms to take indicator variables as input
        for training data labels.
        @return: tuple ( IV_matrix, unique_labels )
        '''
        RC = svo_util.indicator_vars(self.labels)
        if include_label_set:
            return RC
        else:
            return RC[0]
        

if __name__ == '__main__':
    pass