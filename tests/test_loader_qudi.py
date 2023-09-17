from pleasant.loaders.qudi import get_descriptions_in_folder, load_qudi_folder


def test_get_descriptions_in_folder(datadir):
    descriptions = get_descriptions_in_folder(datadir)
    assert len(descriptions) == 1


def test_load_qudi_folder(datadir):
    measurements = load_qudi_folder(str(datadir), break_duration=0.15)
    assert len(measurements) == 6


def test_load_qudi_folder_description_contains(datadir):
    measurements = load_qudi_folder(str(datadir), description_contains="foo")
    assert len(measurements) == 0
