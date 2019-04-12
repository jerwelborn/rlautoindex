# TODO remove this
import os, sys 
head, tail = os.path.split(__file__)
sys.path.insert(0, os.path.join(head, '../../..')) # lift
from lift.lift.rl_model.system_environment import SystemEnvironment
sys.path.insert(0, os.path.join(head, '..')) # /src for access to src/common (though not required for this file)
from common.tpch_util import tpch_tables

import time
if sys.version_info[0] == 2: # opentuner runs 2 not 3
    time.monotonic = time.time 

import enum
import os
import logging 
import psycopg2 as pg




# TODO stick somewhere else
TPCH_DIR = '/local/scratch/jw2027/tpch-tool' 
TPCH_TOOL_DIR = os.path.join(TPCH_DIR, 'dbgen')
DATA_DIR = '/tmp/tables'
DSN = "dbname=postgres user=jw2027"
TPCH_DSN = "dbname=tpch user=jw2027"

class Action(enum.IntEnum):
    noop, duplicate_index, index = 0, 1, 2

class PostgresSystemEnvironment(SystemEnvironment):
    """
        Encapsulates environment

        N.b. agent indices have '_42'


    """

    def __init__(self, tables):
        """
        """
        self.tables = tables

        self.logger = logging.getLogger(__name__)

        self.cxn = self.__connect(DSN)
        try: 
            self.tpch_cxn = self.__connect(TPCH_DSN)
        except pg.OperationalError as e:
            self.tpch_cxn = None 

        self.index_set = set()

    def __connect(self, DSN):
        cxn = pg.connect(DSN)
        cxn.set_session(autocommit=True)
        return cxn
    
    def close(self):
        self.cxn.close()
        self.tpch_cxn.close()

    def act(self, action):
        """
        Creates compound index, as advised by agent

        Args:
            action (dict): contains cols, table containing cols for index
        """
        start = time.monotonic() 
        action_type = None

        cols, tbl = action['index'], action['table']
        index = '_'.join(cols) + '_42'
        if index in self.index_set:
            self.logger.info('action cannot complete (index already in index set)')
            action_type = Action.duplicate_index.name
        elif cols == []:
            self.logger.info('action cannot complete (is no-op)') 
            action_type = Action.noop.name
        else:
            with self.tpch_cxn.cursor() as curs:
                try:
                    self.logger.info("creating compound index %s on %s" % (index, tbl))
                    curs.execute("CREATE INDEX %s ON %s (%s)" %
                                (index, tbl, ','.join(cols)))            
        
                    self.index_set.add(index)
                    action_type = Action.index.name
                except pg.Error as e:
                    print(e)

        return time.monotonic() - start, action_type
    
    def execute(self, query_string, query_string_args=None):
        """
        Having created compound index, executes query and returns runtime

        Args:
            query_string (str)
            query_string_args (tuple)
        """
        runtime = None
        try: 
            with self.tpch_cxn.cursor() as curs:
                start = time.monotonic()

                curs.execute(query_string % query_string_args if query_string_args else query_string)
                runtime = time.monotonic() - start
                curs.fetchall() # TODO 
        except pg.Error as e:
            print(e)

        return runtime

    def system_status(self):
        """
        Compute size of index set
        There are a few approaches for this:
            - the psql command / meta-command \di+ summarizes what we want. Starting psql with psql -E exposes the SQL.
            - see scripts/tpch.py for what I was employing earlier

        Here's how result set is returned:

             Name      |  Table   |   Size
        ---------------+----------+----------
         part_pkey     | part     |  2260992
         region_pkey   | region   |    16384
        ...
        
        [('part_pkey', 'part', 2260992), ('region_pkey', 'region', 16384), ...
        """

        query = """SELECT c.relname as "Name",
                          c2.relname as "Table",
                          pg_catalog.pg_table_size(c.oid) as "Size"
                   FROM pg_catalog.pg_class c
                        LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                        LEFT JOIN pg_catalog.pg_index i ON i.indexrelid = c.oid
                        LEFT JOIN pg_catalog.pg_class c2 ON i.indrelid = c2.oid
                    WHERE c.relkind IN ('i','')
                        AND n.nspname <> 'pg_catalog'
                        AND n.nspname <> 'information_schema'
                        AND n.nspname !~ '^pg_toast'
                        AND pg_catalog.pg_table_is_visible(c.oid);"""
        
        
        with self.tpch_cxn.cursor() as curs:
            curs.execute(query)
            res = curs.fetchall()

        index_set_size = 0.0
        for row in res:
            if '_42' in row[0]: index_set_size += row[2] 
        index_set_size /= 1024*1024 
        
        return index_set_size, self.index_set


    def reset(self):
        """
           Remove all agent-initiated indices 

            schemaname | tablename |   indexname   | tablespace |
           ------------+-----------+---------------+------------+
            public     | customer  | customer_pkey |            |
        """

        table_idxs = 'SELECT * FROM pg_indexes WHERE tablename = \'%s\''
        
        for table in tpch_tables: # TODO self.tables would suffice
        
            with self.tpch_cxn.cursor() as curs:
                curs.execute(table_idxs % table)
                idxs = curs.fetchall()
    
                for idx in idxs:
                    if '_42' in idx[2]:
                        curs.execute('DROP INDEX %s' % idx[2])

        self.index_set.clear()

    # TODO methods to manipulate indices, tpch data, tpch data schema, etc.
    

    
