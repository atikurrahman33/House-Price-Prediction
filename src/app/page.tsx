"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Home, TrendingUp, BarChart3, DollarSign, MapPin, Users, Calendar } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface PredictionResult {
  predicted_price: number
  confidence_interval: [number, number]
  feature_contributions: Array<{
    feature: string
    contribution: number
  }>
  model_performance: {
    test_r2: number
    test_rmse: number
    test_mae: number
  }
}

interface HouseFeatures {
  avg_area_income: number
  avg_area_house_age: number
  avg_area_number_of_rooms: number
  avg_area_number_of_bedrooms: number
  area_population: number
}

export default function HousePricePrediction() {
  const [features, setFeatures] = useState<HouseFeatures>({
    avg_area_income: 68583.11,
    avg_area_house_age: 5.68,
    avg_area_number_of_rooms: 7.01,
    avg_area_number_of_bedrooms: 4.09,
    area_population: 36882.16,
  })

  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleInputChange = (field: keyof HouseFeatures, value: string) => {
    const numValue = Number.parseFloat(value) || 0
    setFeatures((prev) => ({
      ...prev,
      [field]: numValue,
    }))
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)

    const requiredFields = [
      "avg_area_income",
      "avg_area_house_age",
      "avg_area_number_of_rooms",
      "avg_area_number_of_bedrooms",
      "area_population",
    ]

    const missingFields = requiredFields.filter((field) => !features[field] || features[field] <= 0)

    if (missingFields.length > 0) {
      setError(`Please fill in all required fields: ${missingFields.map((f) => f.replace(/_/g, " ")).join(", ")}`)
      setLoading(false)
      return
    }

    try {
      console.log("Sending features:", features)

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(features),
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || "Failed to get prediction")
      }

      setPrediction(result)
    } catch (err) {
      console.error("Prediction error:", err)
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setLoading(false)
    }
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat("en-US").format(num)
  }

  const sampleData = [
    {
      name: "Suburban Family Home",
      data: {
        avg_area_income: 79545.46,
        avg_area_house_age: 5.68,
        avg_area_number_of_rooms: 7.01,
        avg_area_number_of_bedrooms: 4.09,
        area_population: 23086.8,
      },
    },
    {
      name: "Urban Apartment Area",
      data: {
        avg_area_income: 61480.56,
        avg_area_house_age: 7.85,
        avg_area_number_of_rooms: 6.24,
        avg_area_number_of_bedrooms: 3.56,
        area_population: 40173.07,
      },
    },
    {
      name: "Luxury Neighborhood",
      data: {
        avg_area_income: 95000.0,
        avg_area_house_age: 3.2,
        avg_area_number_of_rooms: 8.5,
        avg_area_number_of_bedrooms: 5.2,
        area_population: 18500.0,
      },
    },
  ]

  const loadSampleData = (sampleFeatures: HouseFeatures) => {
    setFeatures(sampleFeatures)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Home className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">Real Estate Price Predictor</h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Advanced machine learning model trained on real housing data. Get accurate price predictions based on area
            demographics and property characteristics.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Area & Property Details
              </CardTitle>
              <CardDescription>Enter the area demographics and property characteristics</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Tabs defaultValue="demographics" className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="demographics">Demographics</TabsTrigger>
                  <TabsTrigger value="property">Property</TabsTrigger>
                </TabsList>

                <TabsContent value="demographics" className="space-y-4">
                  <div>
                    <Label htmlFor="avg_area_income" className="flex items-center gap-2">
                      <DollarSign className="h-4 w-4" />
                      Average Area Income ($)
                    </Label>
                    <Input
                      id="avg_area_income"
                      type="number"
                      min="20000"
                      max="200000"
                      step="100"
                      value={features.avg_area_income}
                      onChange={(e) => handleInputChange("avg_area_income", e.target.value)}
                      placeholder="e.g., 68583"
                    />
                    <p className="text-xs text-gray-500 mt-1">Average household income in the area</p>
                  </div>

                  <div>
                    <Label htmlFor="area_population" className="flex items-center gap-2">
                      <Users className="h-4 w-4" />
                      Area Population
                    </Label>
                    <Input
                      id="area_population"
                      type="number"
                      min="5000"
                      max="100000"
                      step="100"
                      value={features.area_population}
                      onChange={(e) => handleInputChange("area_population", e.target.value)}
                      placeholder="e.g., 36882"
                    />
                    <p className="text-xs text-gray-500 mt-1">Total population in the area</p>
                  </div>

                  <div>
                    <Label htmlFor="avg_area_house_age" className="flex items-center gap-2">
                      <Calendar className="h-4 w-4" />
                      Average House Age (years)
                    </Label>
                    <Input
                      id="avg_area_house_age"
                      type="number"
                      min="0"
                      max="50"
                      step="0.1"
                      value={features.avg_area_house_age}
                      onChange={(e) => handleInputChange("avg_area_house_age", e.target.value)}
                      placeholder="e.g., 5.68"
                    />
                    <p className="text-xs text-gray-500 mt-1">Average age of houses in the area</p>
                  </div>
                </TabsContent>

                <TabsContent value="property" className="space-y-4">
                  <div>
                    <Label htmlFor="avg_area_number_of_rooms">Average Number of Rooms</Label>
                    <Input
                      id="avg_area_number_of_rooms"
                      type="number"
                      min="3"
                      max="15"
                      step="0.1"
                      value={features.avg_area_number_of_rooms}
                      onChange={(e) => handleInputChange("avg_area_number_of_rooms", e.target.value)}
                      placeholder="e.g., 7.01"
                    />
                    <p className="text-xs text-gray-500 mt-1">Average total rooms per house in the area</p>
                  </div>

                  <div>
                    <Label htmlFor="avg_area_number_of_bedrooms">Average Number of Bedrooms</Label>
                    <Input
                      id="avg_area_number_of_bedrooms"
                      type="number"
                      min="1"
                      max="8"
                      step="0.1"
                      value={features.avg_area_number_of_bedrooms}
                      onChange={(e) => handleInputChange("avg_area_number_of_bedrooms", e.target.value)}
                      placeholder="e.g., 4.09"
                    />
                    <p className="text-xs text-gray-500 mt-1">Average bedrooms per house in the area</p>
                  </div>
                </TabsContent>
              </Tabs>

              <div className="border-t pt-4">
                <Label className="text-sm font-medium mb-2 block">Quick Load Sample Data:</Label>
                <div className="grid grid-cols-1 gap-2">
                  {sampleData.map((sample, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => loadSampleData(sample.data)}
                      className="text-left justify-start"
                    >
                      {sample.name}
                    </Button>
                  ))}
                </div>
              </div>

              <Button onClick={handlePredict} disabled={loading} className="w-full" size="lg">
                {loading ? "Predicting..." : "Get Price Prediction"}
              </Button>

              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>

          <Card className="shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Prediction Results
              </CardTitle>
              <CardDescription>AI-powered house price estimation based on real market data</CardDescription>
            </CardHeader>
            <CardContent>
              {prediction ? (
                <div className="space-y-6">
                  <div className="text-center p-6 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border">
                    <div className="flex items-center justify-center gap-2 mb-2">
                      <DollarSign className="h-6 w-6 text-green-600" />
                      <h3 className="text-lg font-semibold text-gray-700">Predicted House Price</h3>
                    </div>
                    <div className="text-4xl font-bold text-green-600 mb-2">
                      {formatPrice(prediction.predicted_price)}
                    </div>
                    <div className="text-sm text-gray-600">
                      Confidence Range: {formatPrice(prediction.confidence_interval[0])} -{" "}
                      {formatPrice(prediction.confidence_interval[1])}
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h4 className="font-semibold mb-3 flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Model Performance
                    </h4>
                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-3 bg-blue-50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {(prediction.model_performance.test_r2 * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Accuracy (R²)</div>
                      </div>
                      <div className="text-center p-3 bg-purple-50 rounded-lg">
                        <div className="text-lg font-bold text-purple-600">
                          {formatPrice(prediction.model_performance.test_rmse)}
                        </div>
                        <div className="text-xs text-gray-600">RMSE</div>
                      </div>
                      <div className="text-center p-3 bg-orange-50 rounded-lg">
                        <div className="text-lg font-bold text-orange-600">
                          {formatPrice(prediction.model_performance.test_mae)}
                        </div>
                        <div className="text-xs text-gray-600">Avg Error</div>
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div>
                    <h4 className="font-semibold mb-3">Feature Impact on Price</h4>
                    <div className="space-y-2">
                      {prediction.feature_contributions.slice(0, 5).map((feature, index) => (
                        <div key={feature.feature} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                          <span className="text-sm font-medium capitalize">
                            {feature.feature.replace(/_/g, " ").replace("avg area", "Area").replace("area", "Area")}
                          </span>
                          <Badge variant={feature.contribution > 0 ? "default" : "secondary"}>
                            {feature.contribution > 0 ? "+" : ""}
                            {formatPrice(feature.contribution)}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-semibold mb-2">Your Input Summary</h4>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>Area Income: {formatPrice(features.avg_area_income)}</div>
                      <div>Population: {formatNumber(features.area_population)}</div>
                      <div>House Age: {features.avg_area_house_age} years</div>
                      <div>Avg Rooms: {features.avg_area_number_of_rooms}</div>
                      <div>Avg Bedrooms: {features.avg_area_number_of_bedrooms}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  <Home className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p className="mb-2">Enter area demographics and property details</p>
                  <p className="text-sm">Click "Get Price Prediction" to see results</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <Card className="mt-8 shadow-lg">
          <CardHeader>
            <CardTitle>About This Real Estate Model</CardTitle>
            <CardDescription>
              Understanding the house price prediction system trained on real market data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Model Architecture
                </h4>
                <p className="text-sm text-gray-600">
                  Linear Regression with feature engineering and standardization. Trained on real housing market data
                  with demographic and property characteristics.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <MapPin className="h-4 w-4" />
                  Data Features
                </h4>
                <p className="text-sm text-gray-600">
                  5 core features: area income, population, house age, room count, and bedroom count. Additional
                  engineered features for better predictions.
                </p>
              </div>
              <div>
                <h4 className="font-semibold mb-2 flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" />
                  Prediction Accuracy
                </h4>
                <p className="text-sm text-gray-600">
                  Model achieves high accuracy on test data with comprehensive validation. Confidence intervals provide
                  prediction uncertainty estimates.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
