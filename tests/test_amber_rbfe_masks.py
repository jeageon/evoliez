from evoliez.adapters.amber_rbfe import _parse_softcore_masks, mdin_ti_min


def test_softcore_mask_parser_accepts_colon_and_bare_masks():
    text = """
 Second stage - Alchemy step for wt.prmtop:
    icfe=1, ifsc=1,
    scmask=':84@HA2,HA3',
 Second stage - Alchemy step for mut.SC.prmtop:
    icfe=1, ifsc=1,
    scmask='85@CB,HB1,HB2',
    """
    masks = _parse_softcore_masks(text)
    assert masks.timask1 == ":84@HA2,HA3"
    assert masks.timask2 == ":85@CB,HB1,HB2"
    deck = mdin_ti_min(masks, 0.5, maxcyc=10)
    assert "scmask1=':84@HA2,HA3'" in deck
    assert "scmask2=':85@CB,HB1,HB2'" in deck


def test_softcore_mask_parser_allows_one_empty_region():
    text = """
 Second stage - Alchemy step for wt.prmtop:
    scmask='',
 Second stage - Alchemy step for mut.SC.prmtop:
    scmask=':10@CB,HB1,HB2',
    """
    masks = _parse_softcore_masks(text)
    assert masks.timask1 == ""
    assert masks.timask2 == ":10@CB,HB1,HB2"
    assert masks.noshakemask == ":10"
