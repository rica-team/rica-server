from typing import Dict

from rica.utils.server import RiCA

prompt_1 = """You're RiCA """

prompt_2 = """, and you're powered by """

prompt_3 = (
    """.

Be different from any others, you're allowed to call tools and still thinking
when waiting for toll callback.

IMPORTANTLY, only generated content in <rica> block will be process by RiCA
Server and any other will be ignored as"""
    """ your thinking which will never be shown to the user.

Here are the tools you can use:"""
)

prompt_4 = """Here is the guidance to call a tool:

The tools are defined in a nested structure. First you have the application,
then the route within that application.

For example, to call the `/exec` route from the `com.example.pyexec` application, you would write:
<rica package="com.example.pyexec" route="/exec">{"code": "1+1"}</rica>

The definitions show which routes can be run in the background and their timeouts.
"""


async def _rica_prompt(apps: Dict[str, RiCA], model_name: str, model_modal: str):
    tools_text = ""
    for package_name, app in apps.items():
        app_desc = app.description or ""
        tools_text += f'<app package="{package_name}" description="{app_desc}">\n'

        for route in app.routes:
            route_desc = route.function.__doc__ or ""
            tools_text += f'  <route path="{route.route}">\n'
            tools_text += f"    <description>{route_desc}</description>\n"
            tools_text += f"    <background>{str(route.background)}</background>\n"
            if route.background:
                tools_text += f"    <timeout>{route.timeout}</timeout>\n"
            tools_text += "  </route>\n"

        tools_text += "</app>\n"

    return prompt_1 + model_name + prompt_2 + model_modal + prompt_3 + tools_text + prompt_4
