from backgammon.board import ImmutableBoard
from backgammon.types import Player
from typing import List


class GameRenderer:
    def __init__(self, board: ImmutableBoard, human_player: Player):
        self.board = board
        self.human_player = human_player
        self.ai_player = (
            Player.PLAYER2 if human_player == Player.PLAYER1 else Player.PLAYER1
        )
        self.TOKEN = {self.human_player: "H", self.ai_player: "A"}

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported")

        # Extract points and colors based on the ImmutableBoard structure
        points = []
        colors = []
        for i in range(24):
            p0 = self.board.positions_0[i]
            p1 = self.board.positions_1[i]
            if p0 > 0 and p1 == 0:
                points.append(p0)
                colors.append(self.TOKEN.get(Player.PLAYER1, "?"))
            elif p1 > 0 and p0 == 0:
                points.append(p1)
                colors.append(self.TOKEN.get(Player.PLAYER2, "?"))
            else:
                points.append(0)
                colors.append(" ")

        # Split the board into top and bottom halves
        bottom_board = points[:12][::-1]
        top_board = points[12:]
        bottom_checkers_color = colors[:12][::-1]
        top_checkers_color = colors[12:]

        assert (
            len(bottom_board) + len(top_board) == 24
        ), "Board split error: points count mismatch."
        assert (
            len(bottom_checkers_color) + len(top_checkers_color) == 24
        ), "Color split error: colors count mismatch."

        # Print the board headers
        print(
            "| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |"
        )
        print(
            "|------------Outer Board-------------|     |-----------P={} Home Board----------|     |".format(
                self.TOKEN.get(self.ai_player, "?")
            )
        )

        # Print the top half of the board
        self.print_half_board(
            top_board, top_checkers_color, self.ai_player, reversed_=True
        )
        print(
            "|------------------------------------|     |-----------------------------------|     |"
        )

        # Print the bottom half of the board
        self.print_half_board(
            bottom_board, bottom_checkers_color, self.human_player, reversed_=False
        )
        print(
            "|------------Outer Board-------------|     |-----------P={} Home Board----------|     |".format(
                self.TOKEN.get(self.human_player, "?")
            )
        )
        print(
            "| 11 | 10 | 9  | 8  | 7  | 6  | BAR | 5  | 4  | 3  | 2  | 1  | 0  | OFF |\n"
        )

    def print_half_board(self, half_board, checkers_color, player, reversed_=False):
        # Map player to index in bar and borne_off tuples
        player_index = player.value

        # Determine the maximum number of checkers in this half-board, bar, or borne_off
        max_half = max(half_board) if half_board else 0
        bar_count = self.board.bar[player_index]
        borne_off_count = self.board.borne_off[player_index]
        max_length = max(max_half, bar_count, borne_off_count)

        # Start printing rows for the current half of the board
        for i in range(max_length):
            # Determine the level of checkers to display
            row = []
            for count, color in zip(half_board, checkers_color):
                if count > i:
                    row.append(color)
                else:
                    row.append(" ")

            # Bar and Off sections
            bar = f"{self.TOKEN[player]} " if bar_count > i else "  "
            off = f"{self.TOKEN[player]} " if borne_off_count > i else "  "

            # Construct the full row with properly aligned columns
            row_display = (
                " | ".join(f"{r:^3}" for r in row[:6])
                + f" | {bar:^3} | "
                + " | ".join(f"{r:^3}" for r in row[6:])
                + f" | {off:^3} |"
            )
            print(f"|  {row_display}")


def render(board: ImmutableBoard, human_player: Player, mode: str = "human") -> None:
    if mode != "human":
        raise NotImplementedError("Only 'human' mode is supported")

    # Determine AI player
    ai_player = Player.PLAYER2 if human_player == Player.PLAYER1 else Player.PLAYER1

    # Define token mapping
    TOKEN = {human_player: "H", ai_player: "A"}

    # Extract points and colors based on the ImmutableBoard structure
    points = []
    colors = []
    for i in range(24):
        p0 = board.positions_0[i]
        p1 = board.positions_1[i]
        if p0 > 0 and p1 == 0:
            points.append(p0)
            colors.append(TOKEN.get(Player.PLAYER1, "?"))
        elif p1 > 0 and p0 == 0:
            points.append(p1)
            colors.append(TOKEN.get(Player.PLAYER2, "?"))
        else:
            points.append(0)
            colors.append(" ")

    # Split the board into top and bottom halves
    bottom_board = points[:12][::-1]
    top_board = points[12:]
    bottom_checkers_color = colors[:12][::-1]
    top_checkers_color = colors[12:]

    assert (
        len(bottom_board) + len(top_board) == 24
    ), "Board split error: points count mismatch."
    assert (
        len(bottom_checkers_color) + len(top_checkers_color) == 24
    ), "Color split error: colors count mismatch."

    # Print the board headers
    print("| 12 | 13 | 14 | 15 | 16 | 17 | BAR | 18 | 19 | 20 | 21 | 22 | 23 | OFF |")
    print(
        f"|------------Outer Board-------------|     |-----------P={TOKEN.get(ai_player, '?')} Home Board----------|     |"
    )

    # Print the top half of the board
    print_half_board(
        half_board=top_board,
        checkers_color=top_checkers_color,
        player=ai_player,
        board=board,
        TOKEN=TOKEN,
        reversed_=True,
    )
    print(
        "|------------------------------------|     |-----------------------------------|     |"
    )

    # Print the bottom half of the board
    print_half_board(
        half_board=bottom_board,
        checkers_color=bottom_checkers_color,
        player=human_player,
        board=board,
        TOKEN=TOKEN,
        reversed_=False,
    )
    print(
        f"|------------Outer Board-------------|     |-----------P={TOKEN.get(human_player, '?')} Home Board----------|     |"
    )
    print("| 11 | 10 | 9  | 8  | 7  | 6  | BAR | 5  | 4  | 3  | 2  | 1  | 0  | OFF |\n")


def print_half_board(
    half_board: List[int],
    checkers_color: List[str],
    player: Player,
    board: ImmutableBoard,
    TOKEN: dict,
    reversed_: bool = False,
) -> None:
    # Map player to index in bar and borne_off tuples
    player_index = player.value - 1  # Assuming Player.PLAYER1 = 1, PLAYER2 = 2

    # Determine the maximum number of checkers in this half-board, bar, or borne_off
    max_half = max(half_board) if half_board else 0
    bar_count = board.bar[player_index]
    borne_off_count = board.borne_off[player_index]
    max_length = max(max_half, bar_count, borne_off_count)

    # Start printing rows for the current half of the board
    for i in range(max_length):
        # Determine the level of checkers to display
        row = []
        for count, color in zip(half_board, checkers_color):
            if count > i:
                row.append(color)
            else:
                row.append(" ")

        # Bar and Off sections
        bar = f"{TOKEN[player]} " if bar_count > i else "  "
        off = f"{TOKEN[player]} " if borne_off_count > i else "  "

        # Construct the full row with properly aligned columns
        row_display = (
            " | ".join(f"{r:^3}" for r in row[:6])
            + f" | {bar:^3} | "
            + " | ".join(f"{r:^3}" for r in row[6:])
            + f" | {off:^3} |"
        )
        print(f"|  {row_display}")
