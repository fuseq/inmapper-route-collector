

/**
 * Bir SVG <path> elemanının 'd' özelliğindeki komutları ayrıştırarak
 * koordinat noktalarının bir listesini çıkarır. (Sadece M, m, L, l destekli)
 * @param {string} dAttr - SVG path elemanının 'd' özelliği.
 * @returns {Array<{x: number, y: number}>} Koordinat noktaları dizisi.
 */
function extractPathPoints(dAttr) {
    const points = [];
    if (!dAttr) {
        return points;
    }

    const commands = dAttr.split(/\s+/).filter(Boolean);
    let currentPoint = null;
    let i = 0;

    while (i < commands.length) {
        const cmd = commands[i];
        if (['M', 'm', 'L', 'l'].includes(cmd)) {
            try {
                let x = parseFloat(commands[i + 1]);
                let y = parseFloat(commands[i + 2]);

                if (cmd === 'm' || cmd === 'l') { // Göreceli koordinatlar
                    if (currentPoint) {
                        x += currentPoint.x;
                        y += currentPoint.y;
                    }
                }

                currentPoint = { x, y };
                points.push(currentPoint);
                i += 3;
            } catch (e) {
                // Hatalı komut formatını atla
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    return points;
}

/**
 * Bir yoldaki keskin ve hafif dönüşleri, ardışık segmentler arasındaki açıya göre tanımlar.
 * @param {Array<{x: number, y: number}>} pathPoints - Rota üzerindeki koordinat noktaları.
 * @param {number} [angleThreshold=89] - Keskin dönüş olarak kabul edilecek minimum açı (derece).
 * @returns {{sharpTurns: Array<{point: {x: number, y: number}, angle: number}>, gentleTurns: Array<{point: {x: number, y: number}, angle: number}>}}
 */
function findSharpTurns(pathPoints, angleThreshold = 89) {
    const sharpTurns = [];
    const gentleTurns = [];

    if (pathPoints.length < 3) {
        return { sharpTurns, gentleTurns };
    }

    for (let i = 1; i < pathPoints.length - 1; i++) {
        const a = pathPoints[i - 1]; // Önceki nokta
        const b = pathPoints[i];     // Mevcut dönüş noktası
        const c = pathPoints[i + 1]; // Sonraki nokta

        const v1 = { x: b.x - a.x, y: b.y - a.y };
        const v2 = { x: c.x - b.x, y: c.y - b.y };

        const mag1 = Math.hypot(v1.x, v1.y);
        const mag2 = Math.hypot(v2.x, v2.y);

        if (mag1 === 0 || mag2 === 0) {
            continue;
        }

        const dot = v1.x * v2.x + v1.y * v2.y;
        let cosTheta = dot / (mag1 * mag2);
        cosTheta = Math.max(-1, Math.min(1, cosTheta)); // [-1, 1] aralığına sıkıştır

        const angle = Math.acos(cosTheta) * (180 / Math.PI);

        if (angle >= angleThreshold) {
            sharpTurns.push({ point: b, angle: angle });
        } else {
            gentleTurns.push({ point: b, angle: angle });
        }
    }

    return { sharpTurns, gentleTurns };
}

/**
 * Bir açının önemli bir dönüşü temsil edip etmediğini belirler.
 * @param {number} angle - Derece cinsinden açı.
 * @returns {boolean} Dönüşün anlamlı olup olmadığı.
 */
function isSignificantTurn(angle) {
    if (angle > 180) {
        angle = 360 - angle;
    }
    if (angle > 175) { // Neredeyse düz
        return false;
    }
    if (angle < 30) { // Çok küçük dönüş
        return false;
    }
    return true;
}

/**
 * Dönüşün "sola" mı yoksa "sağa" mı olduğunu belirler.
 * @param {{x: number, y: number}} prevPoint - Önceki nokta.
 * @param {{x: number, y: number}} turnPoint - Dönüş noktası.
 * @param {{x: number, y: number}} nextPoint - Sonraki nokta.
 * @returns {string} "sola" veya "sağa".
 */
function formatAngleDirection(prevPoint, turnPoint, nextPoint) {
    const v1 = { x: turnPoint.x - prevPoint.x, y: turnPoint.y - prevPoint.y };
    const v2 = { x: nextPoint.x - turnPoint.x, y: nextPoint.y - turnPoint.y };

    // Vektörel çarpımın z bileşeni
    const crossProduct = v1.x * v2.y - v1.y * v2.x;

    // SVG koordinat sisteminde Y aşağı doğru arttığı için işaretler terstir.
    return crossProduct < 0 ? "sola" : "sağa";
}

/**
 * Dönüş noktalarına yakın uygun referans noktalarını (anchor) bulur.
 * @param {Array<{x: number, y: number}>} turnPoints - Analiz edilecek dönüş noktaları.
 * @param {Object.<string, {area: number, centroid: {x: number, y: number}}>} anchorPoints - Potansiyel referans noktaları.
 * @param {number} [distanceThreshold=100] - Aranacak maksimum mesafe.
 * @param {number} [minDistanceBetweenAnchors=200] - Seçilen anchor'lar arasındaki minimum mesafe.
 * @returns {Array<Object>} Uygun bulunan anchor'ların listesi.
 */
function findNearbyAnchors(turnPoints, anchorPoints, distanceThreshold = 100, minDistanceBetweenAnchors = 200) {
    let allMatches = [];

    for (const turnPoint of turnPoints) {
        for (const anchorId in anchorPoints) {
            const anchorData = anchorPoints[anchorId];
            const distToAnchor = distance(anchorData.centroid, turnPoint);

            if (distToAnchor < distanceThreshold) {
                allMatches.push({
                    id: anchorId,
                    area: anchorData.area,
                    distanceToTurn: distToAnchor,
                    centroid: anchorData.centroid,
                    turnPoint: turnPoint,
                });
            }
        }
    }

    // Alana göre (azalan) ve mesafeye göre (artan) sırala
    allMatches.sort((a, b) => {
        if (a.area !== b.area) {
            return b.area - a.area;
        }
        return a.distanceToTurn - b.distanceToTurn;
    });

    const selectedAnchors = [];
    const usedCentroids = [];

    for (const match of allMatches) {
        let isTooClose = false;
        for (const usedCentroid of usedCentroids) {
            if (distance(match.centroid, usedCentroid) < minDistanceBetweenAnchors) {
                isTooClose = true;
                break;
            }
        }
        if (!isTooClose) {
            selectedAnchors.push(match);
            usedCentroids.push(match.centroid);
        }
    }

    return selectedAnchors;
}


