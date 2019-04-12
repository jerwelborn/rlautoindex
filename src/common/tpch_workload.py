import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '..')) # /src for access to src/common (though not required for this file)

from common.tpch_util import tpch_tables, tpch_table_columns, column_type_operators, tpch_string_values, \
    random_float, sample_text, tpch_sample_fns
from common.sql_query import SQLQuery
from common.sql_workload import SQLWorkload

import numpy as np

class TPCHWorkload(SQLWorkload):
    
    def __init__(self, spec):
        """
        Encapsulates workload / workload generation from TPCH 
        Based on TPCH dbgen-derived data but not TPCH qgen-derived queries 
        TODO: replace sampler strategy
        Boilerplated from lift/case_studies/{mongodb,mysql}/ 

        Args: 
            spec (dict): spec for workload; see controller for construction
        """
        
        self.tables = spec['tables'] # which relations to sample from?
        self.n_selections = spec['n_selections'] # how many attributes within those relations?
        self.scale_factor = spec['scale_factor'] # required for scaled_range sampling
        

    def generate_query_template(self):
        """
        Sample a simple query like SELECT COUNT(*) FROM _ WHERE _ AND _ AND _
        """
        # sample a table for FROM clause
        tbl = np.random.choice(self.tables)
        query_string = "SELECT COUNT(*) FROM {} WHERE".format(tbl)
        selections = []
        # randomly select 1,2,..., or n_selections query attributes 
        n_selections = np.random.randint(1, self.n_selections+1)	

        # sample columns from table for WHERE clause
        # TODO sample # of query selections?
        tbl_cols = tpch_table_columns[tbl]
        cols = np.random.choice(list(tbl_cols.keys()), 
                                    size=n_selections, 
                                    replace=False)
        tokens = []

        # sample operators
        for i in range(n_selections):
            col = cols[i]
            desc = tbl_cols[col]
            col_type = desc[0]
            col_type_ops = column_type_operators[col_type]
            col_op = np.random.choice(col_type_ops)
            
            tokens.append(col)
            tokens.append(col_op)

            selection = "{} {} '%s'".format(col, col_op)
            selections.append(selection)
        
        if n_selections == 1:
            selection_string = selection
        else:
            selection_string = " AND ".join(selections)
        query_string = "{} {}".format(query_string, selection_string)

        # sample operands (i.e. %s in above)
        def sample():
            sampled_args = []
            for col in cols:
                desc = tbl_cols[col]
                col_type, sample_type = desc[0], desc[1]

                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(tpch_string_values[col])
                elif sample_type == "fixed_range":
                    range_tuple = desc[2]
                    if col_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif col_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = desc[2]
                    scaled_low = range_tuple[0] * self.scale_factor
                    scaled_high = range_tuple[1] * self.scale_factor
                    if col_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif col_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = tpch_sample_fns[col]()
                elif sample_type == "scaled_sample_fn":
                    sample = tpch_sample_fns[col](self.scale_factor)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(col, desc))

                sampled_args.append(sample)

            return tuple(sampled_args)
        
        return SQLQuery(query_string, query_tbl=tbl, query_cols=cols, 
                        sample_fn=sample, tokens=tokens)

    
    def define_demo_queries(self, n_queries):
        """

        Returns:
            list of SQLQuery: queries encapsulated in SQLQuery query
        """

        pass

    def define_train_queries(self, n_queries):

        return [self.generate_query_template() for _ in range(n_queries)]

    def define_test_queries(self, n_queries):

        return [self.generate_query_template() for _ in range(n_queries)]

    def query_from_csv(self, query_csv):
        """
        Restores a serialized representation of a SQLQuery object.
        
        Args:
            query_csv (list): tighly coupled to sql_query::SQLQuery::as_csv_row
                              [self.query_string, self.query_cols, self.query_tbl, index_cols, self.tokens]
        Returns:
            SQLQuery
        """

        query_cols = query_csv[1].split(",")
        query_tbl = query_csv[2]
        index_columns = None if query_csv[3] == "[]" else query_csv[3].split(",")
        tokens = query_csv[4].split(",")
        query_string = "SELECT COUNT(*) FROM {} WHERE".format(query_tbl)

        # operand (attribute operand) + operator
        selections = []
        for i in range(len(query_cols)):
            selection = "{} {} '%s'".format(query_cols[i], tokens[i*2+1])
            selections.append(selection)

        if len(selections) == 1: 
            selection_string = selection
        else:
            selection_string = " AND ".join(selections)
        query_string = "{} {}".format(query_string, selection_string)
        query = SQLQuery(query_string, query_tbl=query_tbl, index_cols=index_columns,
                         query_cols=query_cols, tokens=tokens)


        query = SQLQuery(query_string, query_tbl=query_tbl, query_cols=query_cols, 
                        index_cols=index_columns, tokens=tokens)

        # operands (attribute value operands)
        # TODO refactor 
        def sample():
            sampled_args = []
            for col in query_cols:
                tbl_cols = tpch_table_columns[query_tbl] # add
                desc = tbl_cols[col] # add
                col_type, sample_type = desc[0], desc[1]

                sample = None
                if sample_type == "lookup":
                    sample = np.random.choice(tpch_string_values[col])
                elif sample_type == "fixed_range":
                    range_tuple = desc[2]
                    if col_type == int:
                        sample = np.random.randint(low=range_tuple[0], high=range_tuple[1])
                    elif col_type == float:
                        sample = random_float(low=range_tuple[0], high=range_tuple[1])
                elif sample_type == "scaled_range":
                    range_tuple = desc[2]
                    scaled_low = range_tuple[0] * self.scale_factor
                    scaled_high = range_tuple[1] * self.scale_factor
                    if col_type == int:
                        sample = np.random.randint(low=scaled_low, high=scaled_high)
                    elif col_type == float:
                        sample = random_float(low=scaled_low, high=scaled_high)
                elif sample_type == "text":
                    sample = sample_text()
                elif sample_type == "sample_fn":
                    sample = tpch_sample_fns[col]()
                elif sample_type == "scaled_sample_fn":
                    sample = tpch_sample_fns[col](self.scale_factor)
                else:
                    raise ValueError("No arg sampled for {} with spec {}".format(col, desc))
                    
                sampled_args.append(sample)
            return tuple(sampled_args)

        query.sample_fn=sample
        return query
