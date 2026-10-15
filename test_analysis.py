import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis import ROOT, DATA, GENRES, load_raw, prepare_data, fit_regressions, prediction, prediction_curve, per_ten_points, choose_primary

@pytest.fixture(scope='module')
def study():
    raw=load_raw()
    clean,valid=prepare_data(raw)
    return raw,clean,valid

@pytest.fixture(scope='module')
def models(study):
    return fit_regressions(study[1])


def test_cleaning_and_sample(study):
    raw,data,valid=study
    assert len(raw)==407
    assert len(valid)==406
    assert len(data)==385
    assert data['primary_genre'].value_counts().to_dict()=={'Action':125,'Drama':120,'Comedy':93,'Horror':47}
    assert data[['title','metascore','budget','gross_worldwide']].notna().all().all()
    assert data['budget'].min()>=1000
    assert set(data.release_year.unique())=={2022,2023,2024,2025}
    assert (data.gross_worldwide>0).all()


def test_first_matching_target_genre():
    assert choose_primary("['Thriller', 'Drama', 'Horror']")=='Drama'
    assert choose_primary("['Romance', 'Biography']") is None


def test_ols_matches_supplied_notebook(models):
    s,m,inter=models
    assert s.rsquared==pytest.approx(0.0038174601,rel=1e-6)
    assert s.pvalues['metascore']==pytest.approx(0.2048128097,rel=1e-5)
    assert m.rsquared==pytest.approx(0.5816955173,rel=1e-6)
    assert m.pvalues['metascore']==pytest.approx(0.0007698455,rel=1e-5)
    assert per_ten_points(m.params['metascore'])==pytest.approx(22.85054,rel=1e-5)


def test_calculator_is_fitted_model_and_ci(models):
    _,_,inter=models
    p=prediction(inter,70,50,'Action',2024)
    assert p['lo']<p['usd']<p['hi']
    assert p['usd']==pytest.approx(102897717.87,rel=1e-6)
    curves=prediction_curve(inter,'Action',50,2024)
    assert len(curves)==51
    assert curves['metascore'].iloc[35]==70
    assert curves['usd'].iloc[35]==pytest.approx(p['usd'],rel=1e-9)


def test_invalid_widget_inputs(models):
    _,_,inter=models
    with pytest.raises(ValueError):
        prediction(inter,80,-10,'Action',2024)


def test_bundled_assets_and_tables():
    for name in ('map_a.png','map_b.png','map_c.png'):
        assert (ROOT/'assets'/name).stat().st_size>100_000
    for name in ('ACDT_OLS_comparison.csv','ACDT_genre_interaction.csv','ACDT_DML_results.csv','RF_feature_importance.csv'):
        assert (DATA/name).is_file()
    assert (ROOT/'Executive_Summary.pdf').stat().st_size>1000
