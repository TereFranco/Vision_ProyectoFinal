class Figure:
    def __init__(self, figure_type: str, color_name: str, color_rgb: tuple[int, int, int], upper_color: tuple [int, int, int], lower_color: tuple[int, int, int], n_vertex: int):
        self.figure_type: str = figure_type
        self.color_name: str = color_name
        self.color_rgb: tuple[int, int, int] = color_rgb
        self.upper_color: tuple[int, int, int] = upper_color
        self.lower_color: tuple[int, int, int] = lower_color
        self.n_vertex: int = n_vertex

    def color_within_tolerance(self, other_color: tuple[int, int, int], tolerance: int = 60) -> bool:
        """
        Verifica si el color está dentro de un rango de tolerancia.

        Args:
            other_color (tuple[int, int, int]): Color a comparar.
            tolerance (int): Rango de tolerancia permitido para cada componente RGB.

        Returns:
            bool: True si el color está dentro del rango, False en caso contrario.
        """
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(self.color_rgb, other_color))

    def is_similar(self, other_color_rgb, other_n_vertex) -> bool:
        return (
            self.color_within_tolerance(other_color_rgb) and
            self.n_vertex == other_n_vertex
        )
    
    def __str__(self) -> str:
        return f"Figure({self.figure_type}, {self.color_name}, {self.n_vertex})"