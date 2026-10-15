"""Offline full-code smoke test with a Streamlit API stub.

This executes all Streamlit app Python code and Plotly object creation,
but it cannot certify the browser or actual Streamlit runtime rendering.
"""
import sys
import runpy
from pathlib import Path
from types import ModuleType


class Panel:
    def __enter__(self): return self
    def __exit__(self,*args): return False


class FakeStreamlit(ModuleType):
    def __init__(self):
        super().__init__('streamlit')
        self.sidebar=Panel()
        self.chart_count=0
        self.download_count=0
    def __getattr__(self,name):
        if name in ('cache_data','cache_resource'):
            return lambda **kwargs: lambda f: f
        if name=='columns':
            return lambda num,**kwargs:[Panel() for _ in range(num if isinstance(num,int) else len(num))]
        if name=='tabs':
            return lambda options:[Panel() for _ in options]
        if name=='expander':
            return lambda *args,**kwargs:Panel()
        if name=='multiselect':
            return lambda label,options,default,**kwargs:default
        if name=='select_slider':
            return lambda label,options,value,**kwargs:value
        if name=='selectbox':
            return lambda label,options,index=0,**kwargs:list(options)[index]
        if name=='slider':
            return lambda label,**kwargs:kwargs['value']
        if name=='plotly_chart':
            def plot(fig,**kwargs):
                assert len(fig.data)>0
                self.chart_count+=1
            return plot
        if name=='download_button':
            def download(*args,**kwargs):
                assert len(kwargs['data'])>0
                self.download_count+=1
            return download
        return lambda *args,**kwargs:None


def test_all_app_python_executes_without_streamlit():
    root=Path(__file__).resolve().parents[1]
    fake=FakeStreamlit()
    sys.modules['streamlit']=fake
    sys.path.insert(0,str(root))
    try:
        runpy.run_path(str(root/'app.py'),run_name='__main__')
        assert fake.chart_count>=4
        assert fake.download_count>=9
    finally:
        sys.modules.pop('streamlit',None)
