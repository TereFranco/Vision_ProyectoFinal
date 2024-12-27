class Figure:
    def __init__(self, figure_type: str, color_name: str, color_rgb: tuple[int, int, int], n_vertex: int):
        self.figure_type: str = figure_type
        self.color_name: str = color_name
        self.color_rgb: tuple[int, int, int] = color_rgb
        self.n_vertex: int = n_vertex

    def color_within_tolerance(self, other_color: tuple[int, int, int], tolerance: int = 40) -> bool:
        """
        Verifica si el color está dentro de un rango de tolerancia.

        Args:
            other_color (tuple[int, int, int]): Color a comparar.
            tolerance (int): Rango de tolerancia permitido para cada componente RGB.

        Returns:
            bool: True si el color está dentro del rango, False en caso contrario.
        """
        return all(abs(c1 - c2) <= tolerance for c1, c2 in zip(self.color_rgb, other_color))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Figure):
            return False
        return (
            self.figure_type == other.figure_type and
            self.color_name == other.color_name and
            self.color_within_tolerance(other.color_rgb) and
            self.n_vertex == other.n_vertex
        )
    
    def __str__(self) -> str:
        return f"Figure({self.figure_type}, {self.color_name}, {self.n_vertex})"
    
