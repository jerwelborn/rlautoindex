# TODO this should be absorbed into other tpch_util

import os
 
TPCH_DIR = '/local/scratch/jw2027/tpch-tool' 
TPCH_TOOL_DIR = os.path.join(TPCH_DIR, 'dbgen')
DATA_DIR = '/tmp/tables'
DSN = "dbname=postgres user=jw2027"
TPCH_DSN = "dbname=tpch user=jw2027"


schema = {
            'lineitem':   ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'], 
            'partsupp':   ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'], 
            'region':     ['r_regionkey', 'r_name', 'r_comment'], 
            'orders':     ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'], 
            'nation':     ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'], 
            'part':       ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'], 
            'supplier':   ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment'], 
            'customer':   ['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment']
          }


# per the spec, though this isn't exactly equal to what I see with SELECT COUNT(*)
schema_records = {
                    'lineitem':     6000000,
                    'partsupp':     800000,
                    'region':       5,
                    'orders':       1500000,
                    'nation':       25,
                    'part':         200000,
                    'supplier':     10000,
                    'customer':     15000
                  }

          