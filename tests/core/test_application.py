import pytest
from rica.core.application import RiCA, Route
from rica.exceptions import PackageInvalidError, RouteExistError

def test_rica_initialization():
    """Test RiCA class initialization."""
    app = RiCA("test.package")
    assert app.package == "test.package"
    assert app.description == ""
    assert app.routes == []

def test_rica_invalid_package_name():
    """Test that RiCA raises an error for invalid package names."""
    with pytest.raises(PackageInvalidError):
        RiCA("invalid-package")

def test_route_registration():
    """Test tool function registration."""
    app = RiCA("test.package")

    @app.route("/test_route")
    def test_function():
        pass

    assert len(app.routes) == 1
    route = app.find_route("/test_route")
    assert isinstance(route, Route)
    assert route.route == "/test_route"
    assert route.function.__name__ == "test_function"

def test_duplicate_route_registration():
    """Test that registering a duplicate route raises an error."""
    app = RiCA("test.package")

    @app.route("/duplicate")
    def first_function():
        pass

    with pytest.raises(RouteExistError):
        @app.route("/duplicate")
        def second_function():
            pass
