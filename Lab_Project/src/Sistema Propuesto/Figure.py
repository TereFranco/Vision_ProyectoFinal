import numpy as np

class Figure:
    def __init__(self, figure_type: str, color_name: str, color_rgb: tuple[int, int, int], lower_color: tuple [int, int, int], upper_color: tuple[int, int, int], n_vertex: int):
        self.figure_type: str = figure_type
        self.color_name: str = color_name
        self.color_rgb: tuple[int, int, int] = color_rgb
        self.lower_color: np.array[int, int, int] = lower_color
        self.upper_color: np.array[int, int, int] = upper_color
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

    def get_color_rgb(self) -> tuple:
        return self.color_rgb
    
    def get_color_bgr(self) -> tuple:
        return self.color_rgb[::-1]

    def get_lower_color(self) -> np.array:
        return self.lower_color
    
    def get_upper_color(self) -> np.array:
        return self.upper_color
    
    def __eq__(self, other):
        if isinstance(other, Figure):
            return (self.figure_type == other.figure_type and 
                    self.color_name == other.color_name and 
                    self.color_rgb == other.color_rgb and 
                    np.array_equal(self.lower_color, other.lower_color) and
                    np.array_equal(self.upper_color, other.upper_color) and
                    self.n_vertex == other.n_vertex)
        
        return False
    
    def is_similar(self, other_color_rgb, other_n_vertex) -> bool:
        return (
            self.color_within_tolerance(other_color_rgb) and
            self.n_vertex == other_n_vertex
        )
    
    def __str__(self) -> str:
        return f"Figure({self.figure_type}, {self.color_name}, {self.n_vertex})"