from django import forms

class SLRPredictionForm(forms.Form):
    distance = forms.FloatField(
        label='Distance Driven (km)',
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter distance in kilometers'
        })
    )

class MLRPredictionForm(forms.Form):
    distance = forms.FloatField(
        label='Distance Driven (km)',
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter distance in kilometers'
        })
    )
    
    speed = forms.FloatField(
        label='Average Speed (km/h)',
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter average speed'
        })
    )
    
    vehicle_weight = forms.FloatField(
        label='Vehicle Weight (kg)',
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter vehicle weight'
        })
    )
    
    temperature = forms.FloatField(
        label='Temperature (°C)',
        required=False,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter temperature (optional)'
        })
    )
    
    traffic_density = forms.ChoiceField(
        label='Traffic Density',
        choices=[
            ('low', 'Low'),
            ('medium', 'Medium'),
            ('high', 'High')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )

class StudentPerformanceForm(forms.Form):
    attendance = forms.FloatField(
        label='Attendance Percentage',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter attendance percentage (0-100)'
        })
    )
    
    previous_grades = forms.FloatField(
        label='Previous Grades (Percentage)',
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter previous grades percentage (0-100)'
        })
    )
    
    study_hours = forms.FloatField(
        label='Study Hours per Week',
        min_value=0,
        max_value=168,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter study hours per week (0-168)'
        })
    )
    
    family_background = forms.ChoiceField(
        label='Family Background',
        choices=[
            ('low', 'Low Income'),
            ('medium', 'Medium Income'),
            ('high', 'High Income')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    extracurricular_activities = forms.ChoiceField(
        label='Extracurricular Activities',
        choices=[
            ('none', 'None'),
            ('low', '1-2 activities'),
            ('medium', '3-4 activities'),
            ('high', '5+ activities')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    parent_education = forms.ChoiceField(
        label='Parent Education Level',
        choices=[
            ('high_school', 'High School'),
            ('bachelors', 'Bachelor\'s Degree'),
            ('masters', 'Master\'s Degree'),
            ('phd', 'PhD')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    class Meta:
        fields = ['attendance', 'previous_grades', 'study_hours', 'family_background', 
                 'extracurricular_activities', 'parent_education']

class CarPerformanceForm(forms.Form):
    # Vehicle specifications
    vehicle_weight = forms.FloatField(
        label='Vehicle Weight (kg)',
        min_value=500,
        max_value=5000,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 1500'})
    )
    engine_power = forms.FloatField(
        label='Engine Power (hp)',
        min_value=50,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 150'})
    )
    engine_displacement = forms.FloatField(
        label='Engine Displacement (L)',
        min_value=0.5,
        max_value=10.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 2.0'})
    )
    
    # Driving conditions
    speed = forms.FloatField(
        label='Speed (km/h)',
        min_value=0,
        max_value=200,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 80'})
    )
    road_gradient = forms.FloatField(
        label='Road Gradient (%)',
        min_value=-15,
        max_value=15,
        initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 0 for flat, 5 for uphill'})
    )
    
    # Aerodynamics
    drag_coefficient = forms.FloatField(
        label='Drag Coefficient',
        min_value=0.2,
        max_value=1.0,
        initial=0.3,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 0.3'})
    )
    
    # Model complexity
    polynomial_degree = forms.ChoiceField(
        label='Polynomial Degree',
        choices=[(2, '2 - Simple Curve'), (3, '3 - Complex Curve'), (4, '4 - Detailed Curve')],
        initial=3,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

class TrafficFlowForm(forms.Form):
    time_of_day = forms.ChoiceField(
        label='Time of Day',
        choices=[
            (6, 'Morning (6-9 AM)'),
            (12, 'Noon (11 AM-2 PM)'),
            (17, 'Evening (4-7 PM)'),
            (22, 'Night (9 PM-12 AM)')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Time of day'
        })
    )
    
    day_of_week = forms.ChoiceField(
        label='Day of Week',
        choices=[
            (1, 'Monday'),
            (2, 'Tuesday'),
            (3, 'Wednesday'),
            (4, 'Thursday'),
            (5, 'Friday'),
            (6, 'Saturday'),
            (7, 'Sunday')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Day of week'
        })
    )
    
    weather_condition = forms.ChoiceField(
        label='Weather Condition',
        choices=[
            (1, 'Clear'),
            (2, 'Light Rain'),
            (3, 'Heavy Rain'),
            (4, 'Snow/Ice')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Weather condition'
        })
    )
    
    road_type = forms.ChoiceField(
        label='Road Type',
        choices=[
            (1, 'Highway'),
            (2, 'Main Road'),
            (3, 'Residential')
        ],
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Road type'
        })
    )
    
    polynomial_degree = forms.ChoiceField(
        label='Model Complexity',
        choices=[
            (2, 'Simple'),
            (3, 'Medium - Recommended'),
            (4, 'Complex')
        ],
        initial=3,
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Model complexity'
        })
    )

class KNNPredictionForm(forms.Form):
    square_footage = forms.FloatField(
        label='Square Footage',
        min_value=100,
        max_value=10000,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 1500'
        })
    )
    
    bedrooms = forms.IntegerField(
        label='Bedrooms',
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 3'
        })
    )
    
    bathrooms = forms.FloatField(
        label='Bathrooms',
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 2.5'
        })
    )
    
    year_built = forms.IntegerField(
        label='Year Built',
        min_value=1900,
        max_value=2023,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 2005'
        })
    )
    
    zip_code = forms.CharField(
        label='ZIP Code',
        max_length=10,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g., 90210'
        })
    )
    
    neighborhood_score = forms.IntegerField(
        label='Neighborhood Score (1-10)',
        min_value=1,
        max_value=10,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Rate from 1-10'
        })
    )
    
    k_neighbors = forms.ChoiceField(
        label='Number of Neighbors (k)',
        choices=[(3, '3 - Minimal averaging'), (5, '5 - Balanced'), (7, '7 - More averaging')],
        initial=5,
        widget=forms.Select(attrs={
            'class': 'form-control form-select',
            'aria-label': 'Number of neighbors'
        })
    ) 