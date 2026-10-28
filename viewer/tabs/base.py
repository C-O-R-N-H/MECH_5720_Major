"""Shape every tab in tabs/ follows. See tabs/__init__.py for how tabs get
registered, and CLAUDE.md for how to add a new one."""


class Tab:
    name = "Tab"

    def render(self, state):
        """Called every UI frame while this tab is the selected tab.
        `state` is the shared state.AppState instance."""
        raise NotImplementedError
